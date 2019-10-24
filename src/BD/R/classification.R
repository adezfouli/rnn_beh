d_total = read.csv(unz(description = "../data/BD/choices_diagno.csv.zip", filename = 'choices_diagno.csv'))

subj_diag = d_total[!duplicated(d_total$ID),]

nrow(subj_diag)

model_diag_ = read.csv('../nongit/results/archive/beh/rnn-pred-diag/rnn_diag.csv')

#uccomment this one for GQL
#model_diag_ = read.csv('../nongit/results/archive/beh/gql-ml-pred-diag/gql_diag.csv')


require(plyr)
model_diag = ddply(model_diag_, "id", function(x){x[which.min(x$loss), ]})

model_subj = merge(subj_diag[,c("ID", "diag")], model_diag[,c("id", "model")], by.x = 'ID', by.y='id')

model_subj$model = factor(model_subj$model, levels=c("Healthy", "Depression", "Bipolar"))
model_subj$diag = factor(model_subj$diag, levels=c("Healthy", "Depression", "Bipolar" ))


############# for getting the effect of random performance
subset(model_subj, ID %in% c(
            "s_038_date_1032011_causality_results.mat",
            "s_058_date_20072011_causality_results.mat",
            "s_074_date_10082011_causality_results.mat",
            "s_136_date_16072012_causality_results.mat",
            "s_14_date_5412_causality_results.mat"
))
#############

# numbers in each group
table(model_subj$diag, model_subj$model)

# proportions
prop.table(table(model_subj$diag, model_subj$model), margin=1)

# overall correct classification rate
mean(model_subj$diag == model_subj$model)

n = 34 + 34 + 33

binom.test(22, 34, 34/n)
binom.test(16, 34, 34/n)
binom.test(15, 33, 33/n)

