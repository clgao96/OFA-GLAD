from Model_pyg import *
from utils_pyg import test_eval, loss_func, edge_index_to_adj_batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
import time
import gc

class Detector:
    def __init__(self, train_config, model_config, data, t):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.GNN = GNNet(self.model_config['input_dim'], self.model_config['hidden_dim'],
                           self.model_config['embedding_dim'],
                           self.model_config['num_layers'], drop_1=self.model_config['drop_1'], bn=self.model_config['bn'],
                           norm=self.model_config['norm']).to(self.train_config['device'])
        self.prot = CrossAttn(self.model_config['prompt_dim'], self.model_config['input_dim'], self.model_config['embedding_dim'],
                              self.model_config['drop_2'], self.model_config['lammda'], self.model_config['alpha'],
                              self.model_config['beta']).to(self.train_config['device'])
        self.t =t


    def pretrain(self):
        best_auc = 0.0
        best_test_score_list = None
        best_epoch = 0
        optimizer_gnn = torch.optim.Adam(self.GNN.parameters(), lr=self.model_config['lr1'])
        for e in range(self.train_config['epochs_pre']):
            for didx, train_data in enumerate(self.data['train']):
                loss_epoch = 0
                num_train = 0
                for batch_idx, gdata in enumerate(train_data):
                    self.GNN.train()
                    gdata = gdata.to(self.train_config['device'])
                    max_nodes = int(gdata.max_node[0].item())
                    edge_index_to_adj_batch(gdata, max_nodes)
                    node_embed, recon_feat, recon_adj = self.GNN(gdata) # GAE
                    _, loss_recon_feat = loss_func_feat(gdata.x, recon_feat, gdata.batch)
                    _, loss_recon_adj = loss_func_adj(gdata.adj, recon_adj)
                    loss = loss_recon_adj.mean() + loss_recon_feat.mean()

                    loss_epoch += loss.item() * gdata.adj.shape[0]
                    num_train += gdata.adj.shape[0]

                    optimizer_gnn.zero_grad()
                    loss.backward()
                    optimizer_gnn.step()
                print("【Pre-train】Epoch: %d Data: %s  Train Loss: %f" % (e, self.train_config['traindsets'][didx], loss_epoch / num_train))

        torch.save(self.GNN.state_dict(), self.train_config['model_path'] + 'gnn_{}.pth'.format(self.t))
        print('********** Finish Pre-train stage! ***********')

    def train(self):
        optimizer_prot = torch.optim.Adam(self.prot.parameters(), lr=self.model_config['lr2'])
        best_auc = 0.0
        best_test_score_list = None
        best_epoch = 0
        all_error_distribution = {}

        for e in range(self.train_config['epochs']):
            for didx, train_data in enumerate(self.data['train']):
                loss_epoch = 0
                num_train = 0
                error_dist_list = []
                for batch_idx, gdata in enumerate(train_data):
                    self.prot.train()
                    self.GNN.eval()
                    gdata = gdata.to(self.train_config['device'])
                    edge_index_to_adj_batch(gdata, gdata.max_node[0])

                    node_embed, recon_feat, recon_adj = self.GNN(gdata)
                    loss = self.prot.get_train_loss_error_distribution(recon_feat, recon_adj, gdata, self.model_config['num_prompt'])

                    loss_epoch += loss.item() * gdata.adj.shape[0]
                    num_train += gdata.adj.shape[0]

                    optimizer_prot.zero_grad()
                    loss.backward()
                    optimizer_prot.step()

                print("Epoch: %d Data: %s  Train Loss: %f" % (e, self.train_config['traindsets'][didx], loss_epoch / num_train))

            del node_embed, recon_feat, recon_adj
            gc.collect()
            torch.cuda.empty_cache()

            if e % 5 == 0:
                auc_dict = {}
                ap_dict = {}
                test_score_list = {}
                all_auc = 0.0
                lable = {}
                scores = {}
                self.GNN.eval()
                self.prot.eval()

                for didx, test_data in enumerate(self.data['test']):
                    print(self.train_config['testdsets'][didx])
                    y = []
                    score = []
                    error_dist_list = []
                    for batch_idx, gdata in enumerate(test_data):
                        gdata = gdata.to(self.train_config['device'])
                        edge_index_to_adj_batch(gdata, gdata.max_node[0])
                        labels = gdata.y
                        query_labels = labels[: -self.model_config['num_prompt']]
                        with torch.no_grad():
                            node_embed, recon_feat, recon_adj = self.GNN(gdata)
                            query_scores = self.prot.get_test_score_error_distribution(recon_feat, recon_adj, gdata, self.model_config['num_prompt'])

                            score.append(query_scores)
                            y.append(query_labels)

                            del node_embed, recon_feat, recon_adj
                            gc.collect()
                            torch.cuda.empty_cache()

                    test_score = test_eval(y, score)
                    all_auc += test_score['AUROC']
                    test_data_name = self.train_config['testdsets'][didx]
                    test_score_list[test_data_name] = {'AUROC': test_score['AUROC'], 'AP': test_score['AP'], 'Precision@K': test_score['Precision@K'], 'Recall@K': test_score['Recall@K']}

                for test_data_name, test_score in test_score_list.items():
                    if test_data_name not in auc_dict:
                        auc_dict[test_data_name] = []
                        ap_dict[test_data_name] = []
                    auc_dict[test_data_name].append(test_score['AUROC'])
                    ap_dict[test_data_name].append(test_score['AP'])
                    print(f"Test on {test_data_name}, AUC is {test_score['AUROC']}, AP is {test_score['AP']}, "
                          f"PRE is {test_score['Precision@K']}, REC is {test_score['Recall@K']}")

                if all_auc > best_auc:
                    best_auc = all_auc
                    best_test_score_list = test_score_list
                    best_epoch = e
                    torch.save(self.prot.state_dict(), self.train_config['model_path'] + 'prot_{}.pth'.format(self.t))
                print('######## Trail {}, Best Epoch {}, Best AUC {}, Now AUC {} ########'.format(self.t, best_epoch, best_auc, all_auc))

        return best_test_score_list