import os
from torch.utils.data import TensorDataset, DataLoader

from Domain_Adaptation.GraphDDPM import GraphDDPMAugmentor
from Domain_Adaptation.GraphTransformerClassifier import GraphTransformerClassifier
from Domain_Adaptation.Transformer_feature_extractor import GraphTransformerExtractor
from Domain_Adaptation.diffusion_da import TotalLoss
from Domain_Adaptation.supervised_contrastive_loss import supervised_contrastive_loss
from dataset.create_adj import create_and_preprocess_adj_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from configs import load_config
from sklearn.model_selection import KFold

count = 1
name = 'proposed'


def save_checkpoint(best_acc, model):
    global path, path_d, path_c
    path = '...\\{0}to{1}\\checkpoints\\{2}-{3}.pth'.format(s, t, name, count)
    print('Best Model Saving...')
    model_state_dict_d = model['d'].state_dict()
    model_state_dict_c = model['c'].state_dict()
    model_state_dict_f = model['f'].state_dict()

    torch.save({
        'model_state_dict_d': model_state_dict_d,
        'model_state_dict_c': model_state_dict_c,
        'model_state_dict_f': model_state_dict_f,
        'best_acc':best_acc,
    }, os.path.join('checkpoints', path))


def K_Flod_spilt(K, fold, data, label):
    '''
    :param K: The number of partitions to divide the dataset into. For example, for a 10-fold split, set K=10.
    :param fold: For example, to retrieve the 5th fold, set flod=5.
    '''
    split_train_list = []
    split_test_list = []
    kf = KFold(n_splits=K)
    for train, test in kf.split(data):
        split_train_list.append(train.tolist())
        split_test_list.append(test.tolist())
    train, test = split_train_list[fold], split_test_list[fold]
    return data[train], data[test], label[train], label[test]


def _train(epoch, source_loader, target_loader, model, optimizer, criterion, args):
    model_d = model['d']
    model_c = model['c']
    model_f = model['f']
    cls_losses = 0.
    recon_losses = 0.
    mmd_losses = 0.
    mmd_losses_p = 0.
    f_losses = 0.
    acc = 0.
    total = 0.
    num = 0.

    model_d.train()
    model_c.train()
    model_f.train()

    for (source_data, source_label), (target_data, target_label) in zip(source_loader, target_loader):
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.long().cuda()
            target_data, target_label = target_data.cuda(), target_label.long().cuda()
        num = num + source_data.shape[0]
        source_adj = create_and_preprocess_adj_matrix(source_data).to(source_data.device)
        target_adj = create_and_preprocess_adj_matrix(target_data).to(target_data.device)
        optimizer.zero_grad()
        label0 = torch.zeros(source_data.shape[0]).long()
        label1 = torch.ones(target_data.shape[0]).long()
        cond0 = F.one_hot(label0, num_classes=2).to(source_data.device)
        cond1 = F.one_hot(label1, num_classes=2).to(source_data.device)

        # Perform diffusion generation on the source and target domain data, respectively
        feature_s, feature_s_pooled = model_f(source_data, source_adj)
        feature_t, feature_t_pooled = model_f(target_data, target_adj)
        loss_recon_s = model_d.noise_pred(source_data, source_adj, feature_s, cond0)
        loss_recon_t = model_d.noise_pred(target_data, target_adj, feature_t, cond1)

        loss_mmd = supervised_contrastive_loss(feature_s_pooled, feature_t_pooled, source_label, target_label)

        loss_recon = loss_recon_s + loss_recon_t
        recon_losses += loss_recon
        mmd_losses += 0.5*loss_mmd

        with torch.no_grad():
            cond = [feature_t, F.one_hot(torch.zeros(target_data.shape[0]).long(), num_classes=2).to(target_data.device)]
            tgt_generated = model_d.sample(target_data, target_adj, cond).to(target_data.device)
            tgt_generated = tgt_generated.reshape_as(target_data)

        # Classification
        cls_data = torch.cat([source_data, tgt_generated], dim=0)
        cls_adj = torch.cat([source_adj, target_adj], dim=0)
        cls_label = torch.cat([source_label, target_label], dim=0)
        cls_feature, cls_output = model_c(cls_data, cls_adj)
        _, pred = F.softmax(cls_output, dim=-1).max(1)
        acc += pred.eq(cls_label).sum().item()
        total += cls_label.size(0)
        loss_cls = criterion.ce_loss(cls_output, cls_label)

        cls_losses += loss_cls
        loss = loss_recon + 0.5*loss_mmd + loss_cls
        loss.backward()

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), max_norm=2)
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), max_norm=2)
            torch.nn.utils.clip_grad_norm_(model_f.parameters(), max_norm=2)
        optimizer.step()

    losses = recon_losses + cls_losses + mmd_losses + 0.5*f_losses + mmd_losses_p
    total_loss = losses/num
    print(
        '[{0}][Epoch: {1:4d}], Loss: {2:.4f}, Loss_cls: {3:.4f}, Loss_recon: {4:.4f}, Loss_mmd: {5:.4f}, Loss_mmd_p: {6:.4f}, Loss_f: {7:.4f}, Acc: {8:.2f}, Correct {9} / Total {10}'.format(
            count, epoch, total_loss, cls_losses / num, recon_losses / num, mmd_losses / num, mmd_losses_p / num, f_losses / num, acc / total * 100., acc,
            total))
    return total_loss


def _eval(epoch, target_loader, model, args):
    model_d = model['d'].eval()
    model_c = model['c'].eval()
    model_f = model['f'].eval()

    acc = 0.
    pred_matrix = []
    target_matrix = []
    TP = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    num = 0.
    with torch.no_grad():
        for (target_data, target_label) in target_loader:
            if args.cuda:
                target_data, target_label = target_data.cuda(), target_label.long().cuda()
            target_adj = create_and_preprocess_adj_matrix(target_data).to(target_data.device)
            cond = torch.zeros(target_data.shape[0]).long().to(target_data.device)
            cond = F.one_hot(cond, num_classes=2).to(target_data.device)
            feature, _ = model_f(target_data, target_adj)
            cond = [feature, cond]
            target_generated = model_d.sample(target_data, target_adj, cond)
            target_generated = target_generated.reshape_as(target_data)
            _, target_output = model_c(target_generated, target_adj)
            _, pred = F.softmax(target_output, dim=-1).max(1)

            for i in range(len(pred)):
                pred_matrix.append(pred[i].cpu())
                target_matrix.append(target_label[i].cpu())
            acc += pred.eq(target_label).sum().item()
            num = num + target_data.shape[0]
        matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        TP += matrix[0, 0]
        FN += matrix[0, 1]
        FP += matrix[1, 0]
        TN += matrix[1, 1]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Sen = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        Spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(
            '[{0}][Epoch: {1:4d}], Acc: {2:.3f}, Sen: {3:.4f}, Spe: {4:.4f}, F1: {5:.4f}, BAC: {6:.4f}'.format(
                count, epoch, acc / num*100, Sen, Spe, f1_score, (Sen + Spe) / 2))

    return acc / num*100, Sen, Spe, f1_score, (Sen + Spe) / 2


def _test(epoch, target_loader, model, args):
    model_d = model['d'].eval()
    model_c = model['c'].eval()
    model_f = model['f'].eval()

    acc = 0.
    pred_matrix = []
    target_matrix = []
    output_tsne = []
    target_tsne = []
    score = np.zeros(shape=[1, 2])
    TP = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    num =0.
    with torch.no_grad():
        for idx, (target_data, target_label) in enumerate(target_loader):
            if args.cuda:
                target_data, target_label = target_data.cuda(), target_label.long().cuda()
            target_adj = create_and_preprocess_adj_matrix(target_data).to(target_data.device)
            cond = torch.zeros(target_data.shape[0]).long().to(target_data.device)
            cond = F.one_hot(cond, num_classes=2).to(target_data.device)
            feature, _ = model_f(target_data, target_adj)
            cond = [feature, cond]
            target_generated = model_d.sample(target_data, target_adj, cond)
            target_generated = target_generated.reshape_as(target_data)
            _, target_output = model_c(target_generated, target_adj)
            if idx == 0:
                output_tsne = target_output.cpu().numpy()
                target_tsne = target_label.cpu().numpy()
            else:
                output_tsne = np.vstack([output_tsne, target_output.cpu().numpy()])
                target_tsne = np.concatenate([target_tsne, target_label.cpu().numpy()], axis=0)
            _, pred = F.softmax(target_output, dim=-1).max(1)
            out = F.softmax(target_output, dim=-1)
            if idx == 0:
                score[0][0] = out[0][0].cpu().numpy()
                score[0][1] = out[0][1].cpu().numpy()
            else:
                score = np.vstack([score, out.cpu().numpy()])

            for k in range(len(pred)):
                pred_matrix.append(pred[k].cpu())
                target_matrix.append(target_label[k].cpu())
            acc += pred.eq(target_label).sum().item()
            bio_onehot = np.empty(shape=[0, 2])
            for i, value in enumerate(target_matrix):
                if value == 0:
                    bio_onehot = np.concatenate((bio_onehot, [[1, 0]]), 0)
                if value == 1:
                    bio_onehot = np.concatenate((bio_onehot, [[0, 1]]), 0)
            label = bio_onehot
            num = num + target_data.shape[0]
        matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        TP += matrix[0, 0]
        FN += matrix[0, 1]
        FP += matrix[1, 0]
        TN += matrix[1, 1]

        precision = TP / (TP + FP)
        Sen = TP / (TP + FN)
        recall = TP / (TP + FN)
        Spe = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)
        fpr, tpr, theresholds = metrics.roc_curve(label.ravel(), score.ravel(), pos_label=1, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)
        print(
            '[Epoch: {0:4d}], Acc: {1:.3f}, Sen: {2:.4f}, Spe: {3:.4f}, F1: {4:.4f}, BAC: {5:.4f}, AUC: {6:.4f}'.format(
                epoch, acc / num * 100., Sen, Spe, f1_score, (Sen + Spe) / 2, auc))

    return acc / num * 100., Sen, Spe, f1_score, (Sen + Spe) / 2, auc, fpr, tpr, output_tsne, target_tsne, matrix


def _folder():
    if not os.path.isdir('...\\{0}to{1}'.format(s, t)):
        os.mkdir('...\\{0}to{1}'.format(s, t))
    if not os.path.isdir('...\\{0}to{1}\\checkpoints'.format(s, t)):
        os.mkdir('...\\{0}to{1}\\checkpoints'.format(s, t))


def main(args):
    global count, s, t, name
    '''
    s: Source Data Folder
    t: Target Data Folder
    load_dataset(s, t): Replace this with your own data loading code
    '''
    s = '...'
    t = '...'
    source_data, source_label, target_data, target_label = load_dataset(s, t)
    t1 = int(0.9*target_data.shape[0])
    target_data_train, target_label_train = target_data[:t1], target_label[:t1]
    target_data_test, target_label_test = target_data[t1:], target_label[t1:]
    k = 10
    _folder()   #create result folder
    '''split dataset and train'''
    test_acc = []
    test_sen = []
    test_spe = []
    test_bac = []
    test_f1 = []
    test_auc = []
    count = 1
    for ii in range(k):
        print('model name:{0}'.format(name))
        # Split the data into 10-fold cross-validation sets
        train, val, label_t, label_v = K_Flod_spilt(k, ii, target_data_train, target_label_train)
        source = TensorDataset(source_data, source_label)
        trainset = TensorDataset(train, label_t)
        valset = TensorDataset(val, label_v)
        testset = TensorDataset(target_data_test, target_label_test)
        print('Source:', source_data.shape)
        print('TrainingSet:', train.shape)
        print('ValSet:', val.shape)
        # Create Dataloader
        source_loader = DataLoader(dataset=source, batch_size=args.batch_size, shuffle=False)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(dataset=valset, batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)

        print('{0} fold:'.format(ii + 1))

        # Create model
        model_d = GraphDDPMAugmentor(T=args.T, in_length=target_data.shape[-1], target_length=source_data.shape[-1], hidden_dim=args.hidden_dim)
        model_c = GraphTransformerClassifier(in_dim=target_data.shape[-1])
        model_f = GraphTransformerExtractor(in_dim=target_data.shape[-1])
        model = {'d':model_d, 'c':model_c, 'f':model_f}

        optimizer = optim.Adam([{"params": model['d'].parameters(), "lr": args.lr, 'weight_decay':args.weight_decay},
                                {"params": model['c'].parameters(), "lr": args.lr, 'weight_decay':args.weight_decay},
                                {"params": model['f'].parameters(), "lr": args.lr, 'weight_decay':args.weight_decay}])

        start_epoch = 1
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if args.cuda:
            model['d'] = model['d'].cuda()
            model['c'] = model['c'].cuda()
            model['f'] = model['f'].cuda()

        criterion = TotalLoss()
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

        # Training
        global_acc = 0.
        global_sen = 0.
        global_spe = 0.
        global_f1 = 0.
        global_bac = 0.
        total_loss = []
        for epoch in range(start_epoch, args.epochs + 1):
            loss = _train(epoch, source_loader, train_loader, model, optimizer, criterion, args)
            total_loss.append(loss.detach().cpu())
            best_acc, best_sen, best_spe, best_f1_score, best_bac = _eval(epoch, val_loader, model, args)
            if global_acc < best_acc and best_sen != 0 and best_spe != 0:
                global_acc, global_sen, global_spe, global_f1, global_bac, global_epoch \
                    = best_acc, best_sen, best_spe, best_f1_score, best_bac, epoch
                save_checkpoint(best_acc, model)
            elif global_acc == best_acc:
                if global_bac < best_bac:
                    global_acc, global_sen, global_spe, global_f1, global_bac, global_epoch \
                        = best_acc, best_sen, best_spe, best_f1_score, best_bac, epoch
                    save_checkpoint(best_acc, model)

            lr_scheduler.step()
            print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
        print(
            '[{0}][Acc: {1:.3f}, Sen: {2:.4f}, Spe: {3:.4f}, F1: {4:.4f}, BAC: {5:.4f}]'.format(ii + 1, global_acc,
                                                                                                global_sen,
                                                                                                global_spe,
                                                                                                global_f1,
                                                                                                global_bac))

        checkpoints = torch.load(path, weights_only=True)
        model['d'].load_state_dict(checkpoints['model_state_dict_d'])
        model['c'].load_state_dict(checkpoints['model_state_dict_c'])
        model['f'].load_state_dict(checkpoints['model_state_dict_f'])

        # Testing
        acc, sen, spe, f1_score, bac, auc, fpr, tpr, feature, target, matrix = _test(start_epoch, test_loader, model, args)
        test_acc.append(acc)
        test_sen.append(sen)
        test_spe.append(spe)
        test_f1.append(f1_score)
        test_bac.append(bac)
        test_auc.append(auc)
        count = count + 1

    print(
        'Acc: {0:.3f}±{1:.3f}, Sen: {2:.4f}±{3:.4f}, Spe: {4:.4f}±{5:.4f}, F1: {6:.4f}±{7:.4f}, BAC: {8:.4f}±{9:.4f}, AUC: {10:.4f}±{11:.4f}'.format(
            np.mean(test_acc), np.std(test_acc), np.mean(test_sen), np.std(test_sen), np.mean(test_spe),
            np.std(test_spe), np.mean(test_f1), np.std(test_f1), np.mean(test_bac), np.std(test_bac),
            np.mean(test_auc), np.std(test_auc)))
    plt.clf()

if __name__ == '__main__':
    args = load_config()
    main(args)
