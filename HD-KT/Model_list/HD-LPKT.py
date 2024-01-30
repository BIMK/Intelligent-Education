import torch
import numpy as np
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

class LPKT(nn.Module):

    def __init__(self, model_settings):
        super(LPKT, self).__init__()
        self.dataset = model_settings['dataset']
        self.n_question = model_settings['skill_num']
        self.device = model_settings['device']
        self.n_exercise = model_settings['problem_num']
        self.n_at = model_settings['n_at']
        self.n_it = model_settings['n_it']
        self.q_matrix = self.get_Q_matrix()
        self.d_k = model_settings['d_k']
        self.d_a = model_settings['d_a']
        self.d_e = model_settings['d_e']
        self.dropout_value = model_settings['dropout']
        self.max_length = model_settings['max_length']
        self.at_embed = nn.Embedding(self.n_at + 10, self.d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(self.n_it + 10, self.d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(self.n_exercise + 10, self.d_e)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
        self.linear_1 = nn.Linear(self.d_a + self.d_e + self.d_k, self.d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(4 * self.d_k, self.d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(4 * self.d_k, self.d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * self.d_k, self.d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(self.d_e + self.d_k, self.d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout_value)
        self.loss_function = nn.BCELoss()
        self.is_dropout = True
        self.tau = 100
        self.emb_dropout = nn.Dropout(0.3)
        self.max_seq_length = model_settings['max_length']
        self.stu_n = model_settings['student_num']
        self.low_dim = model_settings['d_k']

        self.dropout1 = nn.Dropout(model_settings['dropout'])
        self.hidden_size = model_settings['d_k'] * 2
        self.embed_size = model_settings['d_k']
        self.attention_linear1 = nn.Linear(self.low_dim, self.low_dim)
        self.attention_linear2 = nn.Linear(2 * self.low_dim, self.low_dim)
        self.attention_linear3 = nn.Linear(2*self.low_dim, 2)
        self.student_emb = nn.Embedding(self.stu_n + 2, self.low_dim)
        self.knowledge_emb = nn.Embedding(2 * model_settings['skill_num'] + 8, self.low_dim,
                                          padding_idx=0)
        self.exer_emb = nn.Embedding(model_settings['problem_num'] + 5, self.low_dim, padding_idx=0)
        self.skill_num = model_settings['skill_num']
        self.rnn1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.seq_level_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 2, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(self.max_seq_length, self.max_seq_length, (1, 2))
        self.vae = VAE(self.low_dim * 2, self.low_dim, self.low_dim).to(
            model_settings['device'])
        self.g = None
        self.gcn_linear=nn.Linear(2*self.low_dim,model_settings['skill_num'])
        self.deno=nn.Linear(model_settings['skill_num'],2)

    def get_Q_matrix(self):
        path = 'data' + '/' + self.dataset + '/' + 'Q_matrix.txt'
        Q_matrix = np.loadtxt(path, delimiter=' ')

        return torch.FloatTensor(Q_matrix).to(self.device)

    def forward(self, log_dict):
        user_id = torch.LongTensor(log_dict['user_id']).to(self.device)
        exercise_seq_tensor = log_dict['problem_seqs_tensor'].to(self.device)
        skill_seq_tensor = log_dict['skill_seqs_tensor'].to(self.device)
        correct_seq_tensor = log_dict['correct_seqs_tensor'].to(self.device)
        answertime_seq_tensor = log_dict['answertime_seqs_tensor'].to(self.device)
        intervaltime_seq_tensor = log_dict['timeinterval_seqs_tensor'].to(self.device)
        seqs_length = log_dict['seqs_length'].to(self.device)
        batch_size = exercise_seq_tensor.size()[0]
        stu_emb = self.student_emb(user_id)
        exer_emb = self.e_embed(exercise_seq_tensor)
        mask_labels = correct_seq_tensor * (correct_seq_tensor > -1).long()
        knowledge_emb = self.knowledge_emb(skill_seq_tensor + self.skill_num * mask_labels)
        mask = (exercise_seq_tensor != 0).float()
        k_ = torch.cat((exer_emb, knowledge_emb), dim=2)
        stu_deno_singal, cl_loss = self.target_exer_discriminator(stu_emb, k_, mask, skill_seq_tensor)
        element_wise_reconstruction_loss, seq_level_score = self.seq_level_vae(k_, seqs_length, mask)
        seq_level_score = seq_level_score[:, :, 1]
        deno_siginal = (1 - (seq_level_score * stu_deno_singal))
        deno_siginal = deno_siginal.unsqueeze(2).expand(-1, -1, self.low_dim)
        deno_siginal = deno_siginal.masked_fill(deno_siginal == -np.inf, 0)
        answertime_embedding = self.at_embed(
                answertime_seq_tensor) * deno_siginal
        intervaltime_embedding = self.it_embed(intervaltime_seq_tensor) * deno_siginal
        exercise_embeding = self.e_embed(
                exercise_seq_tensor) * deno_siginal
        a_data = correct_seq_tensor.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1,
                                                                             self.d_a)
        all_learning = self.linear_1(torch.cat((exercise_embeding, answertime_embedding, a_data), 2))
        pred = self.recurrent_KT(seq_length=seqs_length,
                                     all_learning=all_learning,
                                     exercise_seq_tensor=exercise_seq_tensor,
                                     exer_embedding=exercise_embeding,
                                     intervaltime_embedding=intervaltime_embedding)
        predictions_packed = nn.utils.rnn.pack_padded_sequence(pred[:, 1:],
                                                                   seqs_length.cpu() - 1,
                                                                   batch_first=True)
        predictions = predictions_packed.data
        labels_packed = torch.nn.utils.rnn.pack_padded_sequence(correct_seq_tensor[:, 1:],
                                                                    seqs_length.cpu() - 1,
                                                                    batch_first=True)
        labels = labels_packed.data
        out_dict = {'predictions': predictions, 'labels': labels, 'clloss': cl_loss,
                        'reloss': element_wise_reconstruction_loss}
        return out_dict


    def get_graph(self, g):
        self.g = g


    def target_exer_discriminator(self, q, k, mask, k_id):

        mask1 = mask.unsqueeze(2).expand(-1, -1, 2 * self.low_dim)
        item_seq_emb = self.emb_dropout(k) * mask1
        encoder_item_seq_emb_bi_direction, (encoder_hidden, mm_) = self.rnn1(item_seq_emb)

        rnn1_hidden = int(encoder_item_seq_emb_bi_direction.shape[-1] / 2)
        encoder_item_seq_emb = encoder_item_seq_emb_bi_direction[:, :, :rnn1_hidden] + \
                               encoder_item_seq_emb_bi_direction[:, :, rnn1_hidden:]

        if self.is_dropout:
            q = q.unsqueeze(1).expand(-1, k.size(1), -1)
            q_ = self.dropout1(self.attention_linear1(q))
            k_1 = self.dropout1(self.attention_linear2(encoder_item_seq_emb))
            k_2 = self.dropout1(self.attention_linear2(k))
        else:
            q = q.unsqueeze(1).expand(-1, k.size(1), -1)
            q_ = self.attention_linear1(q)
            k_1 = self.attention_linear2(encoder_item_seq_emb)
            k_2 = self.dropout1(self.attention_linear2(k))


        cl_loss = 0
        k_=torch.cat((k_1,k_2),dim=2)
        q_=torch.repeat_interleave(q_,2,dim=2)
        alpha = torch.sigmoid(self.attention_linear3(torch.tanh(q_ + k_)))
        gumbel_softmax_alpha = F.gumbel_softmax(alpha, tau=self.tau, hard=True)
        mask = mask.unsqueeze(2).expand(-1, -1, 2)
        gumbel_softmax_alpha = gumbel_softmax_alpha.masked_fill(mask == 0, -np.inf)

        return gumbel_softmax_alpha[:, :, 1], cl_loss

    def seq_level_vae(self, item_seq, item_seq_len, mask, train_flag=True):
        mask1 = mask.unsqueeze(2).expand(-1, -1, 2 * self.low_dim)
        item_seq_emb = self.emb_dropout(item_seq) * mask1
        encoder_item_seq_emb_bi_direction, (encoder_hidden, mm_) = self.rnn1(item_seq_emb)
        rnn1_hidden = int(encoder_item_seq_emb_bi_direction.shape[-1] / 2)
        encoder_item_seq_emb = encoder_item_seq_emb_bi_direction[:, :, :rnn1_hidden] + \
                               encoder_item_seq_emb_bi_direction[:, :, rnn1_hidden:]

        x_reconst, mu, log_var = self.vae(encoder_item_seq_emb)
        reconst_loss = F.mse_loss(x_reconst * mask1, encoder_item_seq_emb * mask1, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        element_wise_reconstruction_loss = reconst_loss + kl_div
        concat_shuffled_and_origin = torch.stack((x_reconst, encoder_item_seq_emb), dim=-1)
        concat_shuffled_and_origin = self.conv(concat_shuffled_and_origin)
        concat_shuffled_and_origin = torch.squeeze(concat_shuffled_and_origin)
        concat_shuffled_and_origin = self.emb_dropout(concat_shuffled_and_origin)
        concat_shuffled_and_origin = nn.ReLU(inplace=True)(concat_shuffled_and_origin)
        reconstruct_score = self.seq_level_mlp(concat_shuffled_and_origin).squeeze()
        mask2 = mask.unsqueeze(2).expand(-1, -1, 2)
        reconstruct_score = reconstruct_score * mask2
        gumbel_softmax_reconstruct_score = F.gumbel_softmax(reconstruct_score, tau=self.tau, hard=True)
        gumbel_softmax_reconstruct_score = gumbel_softmax_reconstruct_score.masked_fill(mask2 == 0, -np.inf)

        return element_wise_reconstruction_loss, gumbel_softmax_reconstruct_score

    def recurrent_KT(self, seq_length, all_learning, exercise_seq_tensor, exer_embedding, intervaltime_embedding):
        batch_size = exercise_seq_tensor.size()[0]
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question, self.d_k)).repeat(batch_size, 1, 1).to(
            self.device)
        h_pre = torch.as_tensor(h_pre, dtype=torch.float)
        h_tilde_pre = None
        learning_pre = torch.zeros(batch_size, self.d_k).to(self.device)
        pred = torch.zeros(batch_size, self.max_length).to(self.device)
        for t in range(max(seq_length) - 1):
            e = exercise_seq_tensor[:, t]
            q_e = self.q_matrix[e].view(batch_size, 1, -1)
            it = intervaltime_embedding[:, t]
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            learning = all_learning[:, t]
            IG = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            IG = self.tanh(IG)
            Gamma_l = self.sig(self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1)))
            LG = Gamma_l * ((IG + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre

            h_tilde = self.q_matrix[exercise_seq_tensor[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size,
                                                                                                       self.d_k)
            y = self.sig(self.linear_5(torch.cat((exer_embedding[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred

    def loss(self, outdict):
        predictions = outdict['predictions']
        labels = outdict['labels']
        labels = torch.as_tensor(labels, dtype=torch.float)
        loss = self.loss_function(predictions, labels) + torch.sum(outdict['reloss'], dim=0, keepdim=False)
        return loss


class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
