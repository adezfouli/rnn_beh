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

for (i in c(1:18)){
    g_data[, paste0("key", i) := shift(key, i, fill=0.5), by=.(ID, block)]
    g_data[, paste0("reward", i) := shift(reward, i, fill=0), by=.(ID, block)]
}

g_data = as.data.frame(g_data)
############### model for each group ##############################
model_list = list()
for (g in c('Healthy', 'Bipolar', 'Depression')){

    train_t = subset(g_data, group == g)
    print(sprintf("group: %s, n = %d", g, length(unique(train_t$ID))))

    model_list[[g]] =  glm(key ~ 1 +
                reward1 * key1 +
                reward2 * key2 +
                reward3 * key3 +
                reward4 * key4 +
                reward5 * key5 +
                reward6 * key6 +
                reward7 * key7 +
                reward8 * key8 +
                reward9 * key9 +
                reward10 * key10 +
                reward11 * key11 +
                reward12 * key12 +
                reward13 * key13 +
                reward14 * key14 +
                reward15 * key15 +
                reward16 * key16 +
                reward17 * key17 +
                reward18 * key18,
            data = train_t, family = binomial)
}
###################################################################

test_with_model = function(model, test_data){
    preds = predict(model, test_data, type= "response")
    keys = test_data$key
    accuracy = mean((preds < 0.5) == (keys == 0), na.rm= TRUE)
    nlp = mean(-log((keys == 1) * preds + (keys == 0) * (1-preds)), na.rm = TRUE)
    nlp
}


nlp_id_group = matrix(ncol=4, nrow=101)

all_ids = unique(g_data$ID)
for(cindex in c(1:length(all_ids))){
    cid = all_ids[cindex]
    test_data = subset(g_data, ID == cid)
    cgroup = test_data[1, c("group")]
    train_data = subset(g_data, group == cgroup & ID != cid)

    g_model =  glm(key ~ 1 +
                reward1 * key1 +
                reward2 * key2 +
                reward3 * key3 +
                reward4 * key4 +
                reward5 * key5 +
                reward6 * key6 +
                reward7 * key7 +
                reward8 * key8 +
                reward9 * key9 +
                reward10 * key10 +
                reward11 * key11 +
                reward12 * key12 +
                reward13 * key13 +
                reward14 * key14 +
                reward15 * key15 +
                reward16 * key16 +
                reward17 * key17 +
                reward18 * key18,
            data = train_data, family = binomial)



    nlp_id_group[cindex, 1] = cid

    if (cgroup == 'Healthy'){
        nlp_id_group[cindex, 2] = test_with_model(g_model, test_data)
        nlp_id_group[cindex, 3] = test_with_model(model_list[['Depression']], test_data)
        nlp_id_group[cindex, 4] = test_with_model(model_list[['Bipolar']], test_data)
    }

    if (cgroup == 'Depression'){
        nlp_id_group[cindex, 2] = test_with_model(model_list[['Healthy']], test_data)
        nlp_id_group[cindex, 3] = test_with_model(g_model, test_data)
        nlp_id_group[cindex, 4] = test_with_model(model_list[['Bipolar']], test_data)
    }

    if (cgroup == 'Bipolar'){
        nlp_id_group[cindex, 2] = test_with_model(model_list[['Healthy']], test_data)
        nlp_id_group[cindex, 3] = test_with_model(model_list[['Depression']], test_data)
        nlp_id_group[cindex, 4] = test_with_model(g_model, test_data)
    }

}

nlp_id_group = data.frame(
                            id = nlp_id_group[, 1],
                            Healthy = as.numeric(nlp_id_group[, 2]),
                            Depression = as.numeric(nlp_id_group[, 3]),
                            Bipolar = as.numeric(nlp_id_group[, 4])
)


d_total = read.csv(unz(description = "../data/BD/choices_diagno.csv.zip", filename = 'choices_diagno.csv'))
subj_diag = d_total[!duplicated(d_total$ID),]

require(plyr)
model_diag = ddply(melt(nlp_id_group, id = "id"), "id", function(x){x[which.min(x$value), ]})

model_subj = merge(subj_diag[,c("ID", "diag")], model_diag[,c("id", "variable")], by.x = 'ID', by.y='id')

model_subj$model = factor(model_subj$variable, levels=c("Healthy", "Depression", "Bipolar"))
model_subj$diag = factor(model_subj$diag, levels=c("Healthy", "Depression", "Bipolar" ))

# numbers in each group
table(model_subj$diag, model_subj$model)

# proportions
prop.table(table(model_subj$diag, model_subj$model), margin=1)

# overall correct classification rate
mean(model_subj$diag == model_subj$model)


######## for testing #########################
a= subset(g_data, group == 'Bipolar' & ID != 's_25_date_40512_causality_results.mat')
a= subset(g_data, group == 'Depression')
g_model =  glm(key ~ 1 +
            reward1 * key1 +
            reward2 * key2 +
            reward3 * key3 +
            reward4 * key4 +
            reward5 * key5 +
            reward6 * key6 +
            reward7 * key7 +
            reward8 * key8 +
            reward9 * key9 +
            reward10 * key10 +
            reward11 * key11 +
            reward12 * key12 +
            reward13 * key13 +
            reward14 * key14 +
            reward15 * key15 +
            reward16 * key16 +
            reward17 * key17 +
            reward18 * key18,
        data = a, family = binomial)

b= subset(g_data, ID == 's_25_date_40512_causality_results.mat')

preds = predict(g_model, b, type = 'response')
length(preds)
keys = b$key
accuracy = mean((preds < 0.5) == (keys == 0), na.rm= TRUE)
nlp = mean(-log((keys == 1) * preds + (keys == 0) * (1-preds)), na.rm = TRUE)
length(keys)

nrow(subset(model_subj, diag == 'Healthy' & model == 'Bipolar'))