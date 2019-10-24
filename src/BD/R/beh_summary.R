paper_mode=TRUE

source('R/helper.R')


data_models = list()
model_stats = read.csv("../nongit/results/archive/beh/to_graph_data/subj_stats.csv")
model_stats$src = "SUBJ"
data_models[[1]] = model_stats

model_stats = read.csv("../nongit/results/archive/beh/to_graph_data/lin_onpolicy.csv")
model_stats$src = "LIN"
data_models[[2]] = model_stats

model_stats = read.csv("../nongit/results/archive/beh/to_graph_data/rnn_onpolicy_nogreedy.csv")
model_stats$src = "RNN"
data_models[[3]] = model_stats

model_stats = read.csv("../nongit/results/archive/beh/to_graph_data/ql_ml_onpolicy_stats.csv")
model_stats$src = "QL"
data_models[[4]] = model_stats

model_stats = read.csv("../nongit/results/archive/beh/to_graph_data/qlp_ml_onpolicy__stats.csv")
model_stats$src = "QLP"
data_models[[5]] = model_stats

model_stats = read.csv("../nongit/results/archive/beh/to_graph_data/gql_ml_onpolicy_stats.csv")
model_stats$src = "GQL"
data_models[[6]] = model_stats

require(data.table)
model_stats = data.frame(rbindlist(data_models))
model_stats$uid = paste0(model_stats$id, model_stats$src)

model_stats$group = factor(model_stats$group, levels = rev(levels(model_stats$group)))

model_stats$src = factor(model_stats$src, levels = c("SUBJ", "RNN", "LIN", "GQL", "QLP", "QL"))

# high vs low key presses
high_low = model_stats[, c("uid", "group", "src", "nonbest_sum", "best_sum")]
require(reshape)
require(ggplot2)

high_low$best_percent = high_low$best_sum / (high_low$best_sum + high_low$nonbest_sum)
high_low$nonbest_percent = high_low$nonbest_sum / (high_low$best_sum + high_low$nonbest_sum)

high_low$best_sum = NULL
high_low$nonbest_sum = NULL

high_low = melt(high_low, id= c("uid", "group", "src"), variable_name = "HL")


ggplot(subset(high_low, HL == "best_percent"), aes(x = HL, y = value, fill = group)) +
 stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.1) +
  geom_point(aes(x = as.numeric(HL) + 0.1), color="red4", position=position_jitterdodge(dodge.width=0.9, jitter.width=0.2), alpha=1.0, size=0.5, stroke = 0.05) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1), size=0.5) +
  xlab("") +
  ylim(c(0, 1)) +
  ylab('probability of \n selecting best action') +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor_nox(legend_position="top", legend.direction='horizontal') +
  facet_grid(. ~ src) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 1.0))

ggsave("../doc/graphs/plots/no-presses.pdf",  width=4, height=3, units = "in", device=cairo_pdf)

# stay on same action
model_stats$pstay_reward = model_stats$reward_stays / (model_stats$reward_switches + model_stats$reward_stays)
model_stats$pstay_noreward = model_stats$noreward_stays / (model_stats$noreward_switches + model_stats$noreward_stays)

stay_sw = model_stats[, c("uid", "group", "src", "pstay_reward", "pstay_noreward")]
require(reshape)
stay_sw = melt(stay_sw, id= c("uid", "group", "src"), variable_name = "sw")

ggplot(subset(stay_sw, T), aes(x = sw, y = value, fill = sw)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour, 
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +  
  ylab('stay probability') +
  geom_point(color="red", position=position_nudge(x = -0.3, y = 0), alpha=0.5, size=0.2) +
  geom_line(aes(group=uid), alpha= 0.1, position=position_nudge(x = -0.3, y = 0), size=0.5) +
  scale_fill_brewer(name = "", palette=palette_mode, breaks = c("pstay_reward", "pstay_noreward"), labels = c("Reward", "No reward")) +
  blk_theme_grid_hor_nox(legend_position="right", legend.direction='vertical') +
  facet_grid(group ~ src) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 1.0))

ggsave("../doc/graphs/plots/sw.pdf",  width=4.5, height=3.5, units = "in", device=cairo_pdf)

# cairo_pdf("../doc/graphs/plots/summary_beh.pdf", width=6, height=6, onefile = FALSE)
# require(gridExtra)
# grid.arrange(p1, p2,ncol=2,nrow=1)
# dev.off()


######## some stats for subjects for number of presses ##############
high_low$group = factor(high_low$group, levels = c('Healthy', 'Depression', 'Bipolar'))
library(lmerTest)
library(lme4)
high_low$HL = factor(high_low$HL, levels = c('nonbest_percent', 'best_percent'))
head(high_low)
contrasts(high_low$HL) = c(0,1)


# for subjects
m = lmer(value ~ HL  + (1 |uid), data = subset(high_low, group == 'Bipolar' & src == 'SUBJ' ))
print(summary(m))

m = lmer(value ~ HL  + (1 |uid), data = subset(high_low, group == 'Healthy' & src == 'SUBJ' ))
print(summary(m))

m = lmer(value ~ HL  + (1 |uid), data = subset(high_low, group == 'Depression' & src == 'SUBJ' ))
print(summary(m))

m = lmer(value ~ HL * group  + (1 |uid), data = subset(high_low, group %in% c('Depression', 'Healthy') & src == 'SUBJ' ))
print(summary(m))

m = lmer(value ~ HL * group  + (1 |uid), data = subset(high_low, group %in% c('Bipolar', 'Healthy') & src == 'SUBJ' ))
print(summary(m))

# for model
m = lmer(value ~ HL  + (1 |uid), data = subset(high_low, group == 'Bipolar' & src == 'RNN' ))
print(summary(m))

m = lmer(value ~ HL  + (1 |uid), data = subset(high_low, group == 'Healthy' & src == 'RNN' ))
print(summary(m))

m = lmer(value ~ HL  + (1 |uid), data = subset(high_low, group == 'Depression' & src == 'RNN' ))
print(summary(m))

task_type = read.csv("../data/BD/id_diag_SC.csv")
head(task_type)
task_type$uid = paste0(task_type$ID, "SUBJ")
task_hl = merge(subset(high_low, src == "SUBJ"), task_type, by = "uid")
head(task_hl)
nrow(task_hl)
# wilcox.test(subset(task_hl, HL == "best_percent" & Scanner.Computer=="C" & group == "Bipolar")$value,
#             subset(task_hl, HL == "best_percent" & Scanner.Computer=="S" & group == "Bipolar")$value, alternative = "two.sided")

m = lmer(value ~ Scanner.Computer + (1|group)  , data = subset(task_hl, src == 'SUBJ' & group %in% c("Healthy", "Bipolar") & HL == "best_percent"))
print(summary(m))

head(data)
######## some stats for subjects for number of probability of staying #######
stay_sw$group = factor(stay_sw$group, levels = c('Healthy', 'Depression', 'Bipolar'))
library(lmerTest)
library(lme4)

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Healthy' & src == 'SUBJ' ))
print(summary(m))

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Depression' & src == 'SUBJ' ))
print(summary(m))

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Bipolar' & src == 'SUBJ' ))
print(summary(m))

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Healthy' & src == 'RNN' ))
print(summary(m))

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Depression' & src == 'RNN' ))
print(summary(m))

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Bipolar' & src == 'RNN' ))
print(summary(m))


m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Healthy' & src == 'QLP' ))
print(summary(m))

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Depression' & src == 'QLP' ))
print(summary(m))

m = lmer(value ~ sw  + (1 |uid), data = subset(stay_sw, group == 'Bipolar' & src == 'QLP' ))
print(summary(m))

# task condition (inside scanner/outside scanner)
task_type = read.csv("../data/BD/id_diag_SC.csv")
head(task_type)
task_type$uid = paste0(task_type$ID, "SUBJ")
task_hl = merge(subset(stay_sw, src == "SUBJ"), task_type, by = "uid")
head(task_hl)

nrow(task_hl)

m = lmer(value ~ Scanner.Computer + (1|group)  , data = subset(task_hl, src == 'SUBJ' & group %in% c("Healthy", "Bipolar") & sw == "pstay_noreward"))
print(summary(m))

m = lmer(value ~ Scanner.Computer + (1|group)  , data = subset(task_hl, src == 'SUBJ' & group %in% c("Healthy", "Bipolar") & sw == "pstay_reward"))
print(summary(m))

# wilcox.test(subset(task_hl, sw == "pstay_reward" & Scanner.Computer=="C" & group == "Healthy")$value,
#             subset(task_hl, sw == "pstay_reward" & Scanner.Computer=="S" & group == "Healthy")$value, alternative = "two.sided")

###### old stats ##################################################
subj_stats = subset(model_stats, src == "SUBJ" & group == "Healthy")
t.test(subj_stats$best_sum, subj_stats$nonbest_sum, paired = T)

subj_stats = subset(model_stats, src == "SUBJ" & group == "Depression")
t.test(subj_stats$best_sum, subj_stats$nonbest_sum, paired = T)

subj_stats = subset(model_stats, src == "SUBJ" & group == "Bipolar")
t.test(subj_stats$best_sum, subj_stats$nonbest_sum, paired = T)


subj_stats = subset(high_low, group %in% c("Healthy", "Bipolar") & src == 'SUBJ')
nrow(subj_stats)
print(summary(aov(value ~ HL * group + Error(uid/ HL), data = subj_stats)))

print(summary(anova(m)))
########### graphs for presentation ###############################
paper_mode=FALSE

source('R/helper.R')
high_low = model_stats[, c("uid", "group", "src", "nonbest_sum", "best_sum")]
require(reshape)
require(ggplot2)
high_low = melt(high_low, id= c("uid", "group", "src"), variable_name = "HL")


ggplot(subset(high_low, src == 'SUBJ'), aes(x = HL, y = value / 12, fill = HL)) +
 stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="green", size=0.2) +
  # geom_boxplot() +
  geom_point(color="green", position=position_nudge(x = -0.2, y = 0), alpha=0.5) +
  geom_line(aes(group=uid), alpha= 0.4, position=position_nudge(x = -0.2, y = 0), size=0.5, color='green') +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('responses/block') +
  scale_fill_brewer(name = "", palette=palette_mode, breaks = c("best_sum", "nonbest_sum"), labels = c("High", "Low")) +
  blk_theme_grid_hor_nox() +
  facet_grid(group ~ .) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))

ggsave("../presentation/graphs/no-presses-subj.pdf",  width=4, height=10, units = "in")



# stay on same action
model_stats$pstay_reward = model_stats$reward_stays / (model_stats$reward_switches + model_stats$reward_stays)
model_stats$pstay_noreward = model_stats$noreward_stays / (model_stats$noreward_switches + model_stats$noreward_stays)

stay_sw = model_stats[, c("uid", "group", "src", "pstay_reward", "pstay_noreward")]
require(reshape)
stay_sw = melt(stay_sw, id= c("uid", "group", "src"), variable_name = "sw")

ggplot(subset(stay_sw, src == 'SUBJ'), aes(x = sw, y = value * 100, fill = sw)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('%stay') +
  geom_point(color="green", position=position_nudge(x = -0.2, y = 0), alpha=0.5) +
  geom_line(aes(group=uid), alpha= 0.4, position=position_nudge(x = -0.2, y = 0), size=0.5, color ='green') +
  scale_fill_brewer(name = "", palette=palette_mode, breaks = c("pstay_reward", "pstay_noreward"), labels = c("Reward", "No reward")) +
  blk_theme_grid_hor_nox() +
  facet_grid(group ~ .) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))

ggsave("../presentation/graphs/sw-subj.pdf",  width=5, height=10, units = "in")


ggplot(subset(high_low, T), aes(x = HL, y = value / 12, fill = HL)) +
 stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="green", size=0.2) +
  # geom_boxplot() +
  geom_point(color="green", position=position_nudge(x = -0.2, y = 0), alpha=0.5) +
  geom_line(aes(group=uid), alpha= 0.4, position=position_nudge(x = -0.2, y = 0), size=0.5, color='green') +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('responses/block') +
  scale_fill_brewer(name = "", palette=palette_mode, breaks = c("best_sum", "nonbest_sum"), labels = c("High", "Low")) +
  blk_theme_grid_hor_nox() +
  facet_grid(group ~ src) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))

ggsave("../presentation/graphs/no-presses.pdf",  width=12, height=10, units = "in")



# stay on same action
model_stats$pstay_reward = model_stats$reward_stays / (model_stats$reward_switches + model_stats$reward_stays)
model_stats$pstay_noreward = model_stats$noreward_stays / (model_stats$noreward_switches + model_stats$noreward_stays)

stay_sw = model_stats[, c("uid", "group", "src", "pstay_reward", "pstay_noreward")]
require(reshape)
stay_sw = melt(stay_sw, id= c("uid", "group", "src"), variable_name = "sw")

ggplot(subset(stay_sw, T), aes(x = sw, y = value * 100, fill = sw)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
  xlab("") +
  ylab('%stay') +
  geom_point(color="green", position=position_nudge(x = -0.2, y = 0), alpha=0.5) +
  geom_line(aes(group=uid), alpha= 0.4, position=position_nudge(x = -0.2, y = 0), size=0.5, color ='green') +
  scale_fill_brewer(name = "", palette=palette_mode, breaks = c("pstay_reward", "pstay_noreward"), labels = c("Reward", "No reward")) +
  blk_theme_grid_hor_nox() +
  facet_grid(group ~ src) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))

ggsave("../presentation/graphs/sw.pdf",  width=12, height=10, units = "in")
