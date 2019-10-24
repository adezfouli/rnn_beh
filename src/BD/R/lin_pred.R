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


############
cv_folds = list()

g='Bipolar'
cv_folds[[g]] = list()
for (cv in c(0:32)){
    cv_folds[[g]][[paste0('fold', cv)]] = read.csv(paste0('../nongit/results/archive/beh/gql-ml-cv/', g, '/fold', cv, '/train_test.csv'))
}

g='Depression'
cv_folds[[g]] = list()
for (cv in c(0:33)){
    cv_folds[[g]][[paste0('fold', cv)]] = read.csv(paste0('../nongit/results/archive/beh/gql-ml-cv/', g, '/fold', cv, '/train_test.csv'))
}


g='Healthy'
cv_folds[[g]] = list()
for (cv in c(0:33)){
    cv_folds[[g]][[paste0('fold', cv)]] = read.csv(paste0('../nongit/results/archive/beh/gql-ml-cv/', g, '/fold', cv, '/train_test.csv'))
}

############
output = list()
for (j in c(1:length(f))){
    for (g in names(cv_folds)){
        for (fold in names(cv_folds[[g]])){
            test_train = cv_folds[[g]][[fold]]
            id = subset(test_train, train == 'test')$ID
            print(sprintf("processsing subject id: %s ", id))
            train_d = subset(g_data, ID %in% subset(test_train, train == 'train')$ID)
            formu = paste0("key", "~", f[[j]])


            model = glm(formu, data = train_d, family = binomial)

            test_d = subset(g_data, ID == id)
            preds = predict(model, test_d, type= "response")
            keys = test_d$key

            accuracy = mean((preds < 0.5) == (keys == 0), na.rm= TRUE)

            nlp = mean(-log((keys == 1) * preds + (keys == 0) * (1-preds)), na.rm = TRUE)
            output[[paste0(id, '_', j)]] = data.frame(id = id, acc = accuracy, nlp = nlp, conf = j, fold=fold, group=g)
        }
    }
}
preds = rbindlist(output)

require(plyr)
ddply(preds, c("group", "conf"), function(x){data.frame(acc = mean(x$acc), nlp = mean(x$nlp))})

write.csv(preds, '../nongit/results/archive/beh/lin_cv/accu_allconfs.csv')

write.csv(subset(preds, conf == 7), '../nongit/results/archive/beh/lin_cv/accu.csv')

preds$group = factor(preds$group, levels = c('Healthy', 'Depression', 'Bipolar'))

preds$degree = as.factor(preds$conf - 1)


library(RColorBrewer)
col = brewer.pal(4, "OrRd")[4]

require(ggplot2)
p1 = ggplot(subset(preds, TRUE), aes(x = degree, y = 100 * acc)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2, fill=col ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("J") +
  ylab('%correct') +
  ylim(c(0, 100)) +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(2,2,2,2), rotate_x=TRUE) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
    theme(axis.text.x=element_text(angle=90, hjust=1)) +
  facet_grid(group~.)

p2 = ggplot(subset(preds, TRUE), aes(x = degree, y = nlp)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2, fill=col ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("J") +
  ylab('NLP') +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(2,2,2,2), rotate_x=TRUE) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
    theme(axis.text.x=element_text(angle=90, hjust=1)) +
  facet_grid(group~.)

library(gridExtra)
cairo_pdf("../doc/graphs/plots/lin_model.pdf", width=6, height=4, onefile = FALSE)
grid.arrange(p2, p1,ncol=2,nrow=1)
dev.off()

###### finding best model ###############
mean_preds = aggregate(nlp ~ conf , data=preds, FUN=mean)

mean_preds[which.min(mean_preds$nlp),]

##################### for testing ####################
test_t = subset(g_data, ID == 's_038_date_1032011_causality_results.mat')
train_t = subset(g_data, group == 'Bipolar' & ID != 's_038_date_1032011_causality_results.mat')

length(unique(train_t$ID))

dd =   data.frame(reward1 = 0,
                        reward2 = 0,
                        reward3 = 0,
                        reward4 = 0,
                        reward5 = 0,
                        reward6 = 0,
                        reward7 = 0,
                        key1 = 1,
                        key2 = 1,
                        key3 = 1,
                        key4 = 1,
                        key5 = 1,
                        key6 = 1,
                        key7 = 0)


model = glm(key ~ 1 +
    reward1 * key1 +
    reward2 * key2 +
    reward3  * key3+
    reward4 * key4 +
    reward5 * key5 +
    reward6 * key6 +
    reward7 * key7 ,
data = train_t, family = binomial)
model

write.csv(train_t, 'tesgt.csv')

preds = predict(model, test_t, type= "response")

mean(log(( 1- preds ) * (1 - (as.numeric(test_t$key) - 1)) + ( preds ) * (as.numeric(test_t$key) - 1)))

mean((preds < 0.5) == (test_t$key == 1))

################ not used in the paper and determining whether there is a bias for keys #####
require(lme4)
r_data$key
model = glmer(key ~ 1 + (1|ID), data = r_data, family = binomial)
print(summary(model))