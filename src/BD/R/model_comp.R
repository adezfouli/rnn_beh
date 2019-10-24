paper_mode=TRUE

source('R/helper.R')

library(gridExtra)

rnn_CV = read.csv('../nongit/results/archive/beh/rnn-cv/rnn-cv-evals/accu.csv')
rnn_CV = subset(rnn_CV,
(
(cell == 20 & group == 'Bipolar' & model_iter == 'model-400')
    |
(cell == 10 &  group == 'Depression' & model_iter == 'model-1200')
|
(cell == 10 &  group == 'Healthy' & model_iter == 'model-1100')
))
rnn_CV$model = 'RNN'

ql_CV = read.csv('../nongit/results/archive/beh/ql-ml-cv/ql-ml-cv-evals/accu.csv')
ql_CV$model = NA
ql_CV$model = "QL"

qlp_CV = read.csv('../nongit/results/archive/beh/qlp-ml-cv/qlp-ml-cv-evals/accu.csv')
qlp_CV$model = NA
qlp_CV$model = "QLP"


gql_CV = read.csv('../nongit/results/archive/beh/gql-ml-cv/gql-ml-cv-evals/accu.csv')
gql_CV$model = NA
gql_CV$model = "GQL"


lin_CV = subset(read.csv('../nongit/results/archive/beh/lin_cv/accu_allconfs.csv'), conf == 19)
lin_CV$model = NA
lin_CV$model = "LIN"

graph_data = rbind(
  rnn_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  ql_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  qlp_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  gql_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  lin_CV[, c('acc', 'fold', 'group', 'nlp', 'model')]

)

graph_data$model = factor(graph_data$model, levels = c("RNN", "LIN", "GQL", "QLP", "QL"))
graph_data$group = factor(graph_data$group, levels = c('Healthy', 'Depression', 'Bipolar'))


require(ggplot2)
p1 = ggplot(subset(graph_data, TRUE), aes(x = model, y = 100 * acc, fill = model)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('%correct') +
  ylim(c(0, 100)) +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(2,2,2,2), rotate_x=TRUE) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
  facet_grid(.~group)

p2 = ggplot(subset(graph_data, TRUE), aes(x = model, y = nlp, fill = model)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('NLP') +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(2,2,2,2), rotate_x=TRUE) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
  facet_grid(.~group)


cairo_pdf("../doc/graphs/plots/model_comp.pdf", width=6, height=2, onefile = FALSE)
grid.arrange(p2, p1,ncol=2,nrow=1)
dev.off()

############ statistics ##########################
# comparison of baselines
sdata = subset(graph_data, model %in% c('GQL', 'QLP'))
sdata$model = factor(sdata$model, levels = c('QLP', 'GQL'))
contrasts(sdata$model) = c(0,1)

# each group (replace Healthy for other groups)
library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Healthy' & model %in% c("GQL", "QLP") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Depression' & model %in% c("GQL", "QLP") ))
print(summary(m))


library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Bipolar' & model %in% c("GQL", "QLP") ))
print(summary(m))

# comparison of baselines: LIN vs GQL
sdata = subset(graph_data, model %in% c('GQL', 'LIN'))
sdata$model = factor(sdata$model, levels = c('LIN', 'GQL'))
contrasts(sdata$model) = c(0,1)

# each group (replace Healthy for other groups)
library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Healthy' & model %in% c("GQL", "LIN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Depression' & model %in% c("GQL", "LIN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Bipolar' & model %in% c("GQL", "LIN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, model %in% c("GQL", "LIN") ))
print(summary(m))


# comparison of baselines: LIN vs QLP
sdata = subset(graph_data, model %in% c('QLP', 'LIN'))
sdata$model = factor(sdata$model, levels = c('LIN', 'QLP'))
contrasts(sdata$model) = c(0,1)

# each group (replace Healthy for other groups)
library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Healthy' & model %in% c("QLP", "LIN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Depression' & model %in% c("QLP", "LIN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Bipolar' & model %in% c("QLP", "LIN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, model %in% c("QLP", "LIN") ))
print(summary(m))


# comparison between RNN and LIN
sdata = subset(graph_data, model %in% c('LIN', 'RNN'))
sdata$model = factor(sdata$model, levels = c('RNN', 'LIN'))
contrasts(sdata$model) = c(0,1)

# each group (replace Healthy for other groups)
library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Healthy' & model %in% c("LIN", "RNN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Depression' & model %in% c("LIN", "RNN") ))
print(summary(m))


library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Bipolar' & model %in% c("LIN", "RNN") ))
print(summary(m))

# whole group for GQL
library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, model %in% c("LIN", "RNN") ))
print(summary(m))


# comparison between RNN and gql
sdata = subset(graph_data, model %in% c('GQL', 'RNN'))
sdata$model = factor(sdata$model, levels = c('RNN', 'GQL'))
contrasts(sdata$model) = c(0,1)

summary(subset(sdata, group =='Healthy' & model %in% c("GQL") )$nlp)

# each group (replace Healthy for other groups)
library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Healthy' & model %in% c("GQL", "RNN") ))
print(summary(m))

library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Depression' & model %in% c("GQL", "RNN") ))
print(summary(m))


library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, group =='Bipolar' & model %in% c("GQL", "RNN") ))
print(summary(m))

# whole group for GQL
library(lmerTest)
library(lme4)
m = lmer(nlp ~ model + (1 | fold), data = subset(sdata, model %in% c("GQL", "RNN") ))
print(summary(m))


### for between group hyper -parameter estimation
# for in sample graophs
rnn_CV = read.csv('../nongit/results/archive/beh/rnn_cv/rnn-cv-evals/accu.csv')
rnn_CV = subset(rnn_CV,
(
(cell == 10 & group == 'Bipolar' & model_iter == 'model-1200')
    |
(cell == 10 &  group == 'Depression' & model_iter == 'model-1100')
|
(cell == 10 &  group == 'Healthy' & model_iter == 'model-1200')
))
rnn_CV$model = 'RNN'

ql_CV = read.csv('../nongit/results/archive/beh/ql-ml-cv/ql-ml-cv-evals/accu.csv')
ql_CV$model = NA
ql_CV$model = "QL"

qlp_CV = read.csv('../nongit/results/archive/beh/qlp-ml-cv/qlp-ml-cv-evals/accu.csv')
qlp_CV$model = NA
qlp_CV$model = "QLP"


gql_CV = read.csv('../nongit/results/archive/beh/gql-ml-cv/gql-ml-cv-evals/accu.csv')
gql_CV$model = NA
gql_CV$model = "GQL"


lin_CV = subset(read.csv('../nongit/results/archive/beh/lin_cv/accu_allconfs.csv'), conf == 19)
lin_CV$model = NA
lin_CV$model = "LIN"

graph_data = rbind(
  rnn_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  ql_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  qlp_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  gql_CV[, c('acc', 'fold', 'group', 'nlp', 'model')],
  lin_CV[, c('acc', 'fold', 'group', 'nlp', 'model')]

)

graph_data$model = factor(graph_data$model, levels = c("RNN", "LIN", "GQL", "QLP", "QL"))
graph_data$group = factor(graph_data$group, levels = c('Healthy', 'Depression', 'Bipolar'))


require(ggplot2)
p1 = ggplot(subset(graph_data, TRUE), aes(x = model, y = 100 * acc, fill = model)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('%correct') +
  ylim(c(0, 100)) +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(2,2,2,2), rotate_x=TRUE) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
  facet_grid(.~group)

p2 = ggplot(subset(graph_data, TRUE), aes(x = model, y = nlp, fill = model)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('NLP') +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(2,2,2,2), rotate_x=TRUE) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
  facet_grid(.~group)


cairo_pdf("../doc/graphs/plots/model_comp_other_group.pdf", width=6, height=2, onefile = FALSE)
grid.arrange(p2, p1,ncol=2,nrow=1)
dev.off()

########### for presentation #####################
paper_mode=FALSE

source('R/helper.R')

library(gridExtra)

rnn_CV = read.csv('../nongit/results/archive/beh/BD_RNN_CV/evals/accu.csv')
rnn_CV = subset(rnn_CV, cell == 10 &
(
(group == 'Bipolar' & model_iter == 'model-700')
    |
(group == 'Depression' & model_iter == 'model-1300')
|
(group == 'Healthy' & model_iter == 'model-100')
))
rnn_CV$model = 'RNN'

qrl_CV = read.csv('../nongit/results/archive/beh/BD_QRL_CV/evals/accu.csv')
qrl_CV$model = NA
qrl_CV[qrl_CV$option == 'persv_False-', 'model'] = 'QL'
qrl_CV[qrl_CV$option == 'persv_True-', 'model'] = 'QLP'

graph_data = rbind(rnn_CV[, c('acc', 'fold', 'group', 'nlp', 'model')], qrl_CV[, c('acc', 'fold', 'group', 'nlp', 'model')])

p1 = ggplot(subset(graph_data, TRUE), aes(x = model, y = 100 * acc, fill = model)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('%correct') +
  ylim(c(0, 100)) +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(10,10,10,10)) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
  facet_grid(.~group)

p2 = ggplot(subset(graph_data, TRUE), aes(x = model, y = nlp, fill = model)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('NLP') +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(10,10,10,10)) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
  facet_grid(.~group)


pdf("../presentation/graphs/model_comp.pdf", width=8, height=10, onefile = FALSE, title= '')
grid.arrange(p1, p2,ncol=1,nrow=2)
dev.off()





