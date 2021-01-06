import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from utils import load_data, EarlyStopping
from sampler import sample

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, classifier, g, features, labels, mask, loss_func):
    model.eval()
    classifier.eval()
    with torch.no_grad():
        embeddings = model(g, features)
        logits = classifier(embeddings[0])
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=args['device'])
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    # features = features.to(args['device'])
    features = [f.to(args['device']) for f in features]
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    if args['hetero']:
        from model_hetero import SS_HAN
        model = SS_HAN(muti_meta_paths=
                    [[['pa', 'ap'], ['pf', 'fp']],
                    [['ap', 'pa']],
                    [['fp', 'pf']]],
                    in_size=features[0].shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])

        g = g.to(args['device'])
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]

    stopper = EarlyStopping(patience=args['patience'])
    # loss_fcn = F.binary_cross_entropy_with_logits
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    print('*****************************Pre-training Starting*************************************')
    for epoch in range(args['pretrain_epochs']):
        model.train()

        for idx in range(args['batch_size']):
            embeddings = model(g, features)
            pos_edge_index, neg_edge_index = sample(g, 1)
            link_logits = model.calculate_loss(embeddings, pos_edge_index, neg_edge_index)
            link_labels = get_link_labels(pos_edge_index, neg_edge_index)
            loss = loss_fcn(link_logits, link_labels)
            link_probs = link_logits.sigmoid().detach().numpy()
            acc = roc_auc_score(link_labels, link_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('link_labels : {}'.format(link_labels))
            # print('link_probs : {}'.format(link_probs))
            print('epoch: {} || batch_size : {} || loss: {} || accuracy: {}'.format(epoch, idx, loss, acc))
        # lr_scheduler.step()
        early_stop = stopper.step(model, epoch, loss.item(), acc)
        if early_stop:
            break
    filename = './model/ss-han_{}_{:02f}_{:02f}'.format(epoch, loss, acc)
    torch.save(model.state_dict(), filename)

    print('*****************************Pre-training Ending*************************************')
    print('\n')
    print('*****************************Fine-tuning Starting*************************************')

    # freeze the pretrained parameter
    for parms in model.parameters():
        parms.requires_grad = False

    from model_hetero import Classifier
    classifier = Classifier(in_size=args['hidden_units']*args['num_heads'][-1],
                            hidden_size=128,
                            out_size=num_classes)

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['fine-tuning_epochs']):
        model.train()

        embeddings = model(g, features)
        output = classifier(embeddings[0])
        loss = loss_fcn(output[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(output[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 \
            = evaluate(model, classifier, g, features, labels, val_mask, loss_fcn)
        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

    print('*****************************Fine-tuning Ending*************************************')

    test_loss, test_acc, test_micro_f1, test_macro_f1 \
        = evaluate(model, classifier, g, features, labels, val_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))


if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', default=True,
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)