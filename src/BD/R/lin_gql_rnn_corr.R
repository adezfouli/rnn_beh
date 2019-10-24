############## for LIN model #################
paper_mode=TRUE

source('R/helper.R')
require(plyr)

require(data.table)
r_data = read.csv(unz(description = "../data/BD/choices_diagno.csv.zip", filename = 'choices_diagno.csv'))
r_data$ID = as.character(r_data$ID)
r_data$key = as.character(r_data$key)
r_data[r_data$key == "R1",]$key = 1
r_data[r_data$key == "R2",]$key = 2
r_data$key = as.numeric(r_data$key) - 1
r_data$group = r_data$diag


r_data$reward = 1 * (r_data$outcome != "null")
g_data = data.table(r_data)

for (i in c(1:21)){
    g_data[, paste0("key", i) := shift(key, i, fill=0.5), by=.(ID, block)]
    g_data[, paste0("reward", i) := shift(reward, i, fill=0), by=.(ID, block)]
}

f = list()
K = 20
f[['0']] = "1"
for (k in c(1:K)){
    s = "1"
    for (j in c(1:k)){
        s = paste0(s, ' + ', 'reward', j , '*', 'key', j)
    }
    f[[as.character(k)]] = s
}

l = list()
for (g in c('Healthy', 'Depression', 'Bipolar')){
    formu = paste0("key", "~", f[[19]])
    g_group = subset(g_data, group == g)
    model = glm(formu, data = g_group, family = binomial)
    g_group$preds = predict(model, g_group, type='response')
    l[[g]] = g_group
}

lin_preds = rbindlist(l)

########### for GQL #################################################################################
gql_1 = read.csv(paste0('../nongit/results/archive/beh/gql-ml-opt/Healthy/policies.csv'))
gql_2 = read.csv(paste0('../nongit/results/archive/beh/gql-ml-opt/Depression/policies.csv'))
gql_3 = read.csv(paste0('../nongit/results/archive/beh/gql-ml-opt/Bipolar/policies.csv'))

gql_preds = rbind(gql_1, gql_2, gql_3)
gql_preds$preds = exp(gql_preds$X1)

############# gql and lin correlation analysis in predictions for subjects' data ###################
gql_lin_corr = matrix(ncol=2, nrow=101)

r = 1
for (cid in unique(gql_preds$id)){
    gql_p = subset(gql_preds, id == cid)$preds
    lin_p = subset(lin_preds, ID == cid)$preds
    gql_lin_corr[r, 1] = cid
    gql_lin_corr[r, 2] = a= cor(gql_p,  lin_p, method = 'spearman')
    r = r + 1
}

gql_lin_corr = data.frame(id = gql_lin_corr[, 1], corr = as.numeric(gql_lin_corr[, 2]))
mean(gql_lin_corr$corr)

#################### rnn correlation ##########################################################
rnn_1 = read.csv(paste0('../nongit/results/archive/beh/rnn-on-sims-data/Healthy/policies-.csv'))
rnn_2 = read.csv(paste0('../nongit/results/archive/beh/rnn-on-sims-data/Depression/policies-.csv'))
rnn_3 = read.csv(paste0('../nongit/results/archive/beh/rnn-on-sims-data/Bipolar/policies-.csv'))

rnn_preds = rbind(rnn_1, rnn_2, rnn_3)
rnn_preds$preds = rnn_preds$X1


rnn_lin_corr = matrix(ncol=2, nrow=101)

r = 1
for (cid in unique(rnn_preds$id)){
    rnn_p = subset(rnn_preds, id == cid)$preds
    lin_p = subset(lin_preds, ID == cid)$preds
    rnn_lin_corr[r, 1] = cid
    rnn_lin_corr[r, 2] = cor(rnn_p,  lin_p, method = 'spearman')
    r = r + 1
}

rnn_lin_corr = data.frame(id = rnn_lin_corr[, 1], corr = as.numeric(rnn_lin_corr[, 2]))
mean(rnn_lin_corr$corr)

############# GQL and RNN ###############
rnn_gql_corr = matrix(ncol=2, nrow=101)

r = 1
for (cid in unique(rnn_preds$id)){
    rnn_p = subset(rnn_preds, id == cid)$preds
    gql_p = subset(gql_preds, id == cid)$preds
    rnn_gql_corr[r, 1] = cid
    rnn_gql_corr[r, 2] = cor(rnn_p,  gql_p, method = 'spearman')
    r = r + 1
}

rnn_gql_corr = data.frame(id = rnn_gql_corr[, 1], corr = as.numeric(rnn_gql_corr[, 2]))
mean(rnn_gql_corr$corr)
