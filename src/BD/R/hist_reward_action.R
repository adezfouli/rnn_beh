require(reshape)
paper_mode = TRUE
source('R/helper.R')

draw_action_rew_effect("../nongit/results/archive/beh/to_graph_data/subj_all_data.csv", 'subj')
draw_action_rew_effect("../nongit/results/archive/beh/to_graph_data/rnn_all_data.csv", 'rnn')
draw_action_rew_effect("../nongit/results/archive/beh/to_graph_data/gql_all_data_ml.csv", 'gql')
draw_action_rew_effect("../nongit/results/archive/beh/to_graph_data/gql10d_all_data_ml.csv", 'gql10d')
draw_action_rew_effect("../nongit/results/archive/beh/to_graph_data/lin_all_data.csv", 'lin')

draw_cur_pre_all_models()
draw_cur_pre_all_models = function(){
    oscc_sw1 = read.csv("../nongit/results/archive/beh/to_graph_data/subj_all_data.csv")[, c("ID", "prev_key", "sim_ID", "group", "same")]
    oscc_sw1$src = 'SUBJ'

    oscc_sw2 = read.csv("../nongit/results/archive/beh/to_graph_data/rnn_all_data.csv")[, c("ID", "prev_key", "sim_ID", "group", "same")]
    oscc_sw2$src = 'RNN'

    oscc_sw3 = read.csv("../nongit/results/archive/beh/to_graph_data/lin_all_data.csv")[, c("ID", "prev_key", "sim_ID", "group", "same")]
    oscc_sw3$src = 'LIN'

    oscc_sw4 = read.csv("../nongit/results/archive/beh/to_graph_data/gql_all_data_ml.csv")[, c("ID", "prev_key", "sim_ID", "group", "same")]
    oscc_sw4$src = 'GQL'

    oscc_sw = rbind(oscc_sw1, oscc_sw2, oscc_sw3, oscc_sw4)

    if (!("group" %in% colnames(oscc_sw))){
        oscc_sw$group = NA
        oscc_sw$group = oscc_sw$diag
    }


    if (!("ID" %in% colnames(oscc_sw))){
        oscc_sw$ID = NA
        oscc_sw$ID = oscc_sw$id
    }


    oscc_sw = subset(oscc_sw, same == "False")[, c("ID", "prev_key", "sim_ID", "group", "src")]
    oscc_sw$cur_run = oscc_sw$prev_key + 1
    oscc_sw$prev_key = NULL
    library(data.table)
    oscc_sw$pre_run = shift(oscc_sw$cur_run, )
    oscc_sw$pre_sim_ID = shift(oscc_sw$sim_ID, )
    oscc_sw$pre_run[oscc_sw$pre_sim_ID != oscc_sw$sim_ID ] = NA
    # write.csv(oscc_sw, "output.csv")

    oscc_avg = aggregate(cur_run ~ pre_run + ID + group + src, data = subset(oscc_sw, pre_run < 11), FUN=median)

    oscc_avg$group = factor(oscc_avg$group, levels = rev(levels(oscc_avg$group)))
    oscc_avg$src = as.factor(oscc_avg$src)
    oscc_avg$src = factor(oscc_avg$src, levels = rev(levels(oscc_avg$src)))

    ggplot(subset(oscc_avg, TRUE), aes(y=cur_run, x=as.factor(pre_run))) +
        stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(),  color="black", fill=brewer.pal(4, "OrRd")[2], size=0.2) +
        stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
                   position=position_dodge(.9),  fun.args = list(mult = 1)) +
        # geom_point(color="red", position=position_nudge(x = -0.3, y = 0), alpha=0.3, size=0.2) +
        geom_abline(intercept = 0, slope = 1, linetype='dashed')     +
        xlab("length of previous run") +
        ylab("length of current run") +
        scale_fill_brewer(name = "", palette=palette_mode) +
        blk_theme_no_grid(legend_position = "right", margins = c(2,2,2,2)) +
        # theme(panel.grid.minor.y = element_line(colour = text_color)) +
        facet_grid(group~src)

    ggsave(paste0("../doc/graphs/plots/", "all_models", "_run_length_pre_cur.pdf"),
           width=6, height=4, units = "in", device=cairo_pdf)

}

draw_action_rew_effect = function(input_path, output_prefix){
    rew_key = read.csv(input_path)
    rew_key = subset(rew_key, reward == 1)
    rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
    rew_key$pre_rew_group = NA
    rew_key[rew_key$prev_rewards == 0,]$pre_rew_group = "0"
    rew_key[rew_key$prev_rewards == 1,]$pre_rew_group = "1"
    rew_key[rew_key$prev_rewards == 2,]$pre_rew_group = "2"
    rew_key[rew_key$prev_rewards > 2,]$pre_rew_group = ">2"
    rew_key$same = (rew_key$same  == "True")* 1
    agg_rew_key = aggregate(same ~ pre_rew_group + ID + group, data = rew_key, FUN=mean)

    require(ggplot2)
    library(RColorBrewer)
    col = brewer.pal(4, "OrRd")[2]
    p1 = ggplot(agg_rew_key, aes(y = same, x=pre_rew_group)) +
        stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", fill=col, size=0.2) +
        geom_point(color="red", position=position_nudge(x = -0.2, y = 0), alpha=0.3, size=0.2) +
        stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
                   position=position_dodge(.9),  fun.args = list(mult = 1)) +
        scale_fill_brewer(name = "", palette=palette_mode) +
        xlab("#rewards since  \n switching to the current action") +
        ylim(c(0,1)) +
        ylab('stay probability') +
        blk_theme_no_grid(legend_position = "right", margins = c(2,2,2,2)) +
        facet_grid(.~group)


    key_count_sw = read.csv(input_path)
    key_count_sw$group = factor(key_count_sw$group, levels = rev(levels(key_count_sw$group)))
    key_count_sw$same  = (key_count_sw$same == "True") * 1
    key_count_sw = aggregate(same ~ prev_key + ID+ group, data = subset(key_count_sw, prev_key < 15 & reward == 0 & prev_rewards == 0), FUN=mean)
    p2 = ggplot(subset(key_count_sw, TRUE), aes(y= same, x=prev_key)) +
        stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(),  color="black", fill=brewer.pal(4, "OrRd")[2], size=0.2) +
        stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
                   position=position_dodge(.9),  fun.args = list(mult = 1), size=0.1) +
        geom_smooth(color=brewer.pal(4, "OrRd")[4], linetype ='solid', alpha=0.4, size=0.5) +
        xlab("#actions since  \n switching to the current action") +
        ylab('stay probability') +
        ylim(c(0,1)) +
        scale_fill_brewer(name = "", palette=palette_mode) +
        blk_theme_no_grid(legend_position = "right", margins = c(2,2,2,2)) +
        facet_grid(.~group)

    library(lmerTest)
    library(lme4)
    m = lmer(same ~ prev_key + (1 | ID), data = subset(key_count_sw, group =='Healthy'))
    print(summary(m))


    cairo_pdf(paste0("../doc/graphs/plots/", output_prefix, "_pre_reward_actions.pdf"), width=6, height=2, onefile = FALSE)
    library(gridExtra)
    grid.arrange(p1, p2,ncol=2,nrow=1, widths=c(2.5/5, 2.5/5))
    dev.off()


    oscc_sw = read.csv(input_path)

    if (!("group" %in% colnames(oscc_sw))){
        oscc_sw$group = NA
        oscc_sw$group = oscc_sw$diag
    }


    if (!("ID" %in% colnames(oscc_sw))){
        oscc_sw$ID = NA
        oscc_sw$ID = oscc_sw$id
    }


    oscc_sw = subset(oscc_sw, same == "False")[, c("ID", "prev_key", "sim_ID", "group")]
    oscc_sw$cur_run = oscc_sw$prev_key + 1
    oscc_sw$prev_key = NULL
    library(data.table)
    oscc_sw$pre_run = shift(oscc_sw$cur_run, )
    oscc_sw$pre_sim_ID = shift(oscc_sw$sim_ID, )
    oscc_sw$pre_run[oscc_sw$pre_sim_ID != oscc_sw$sim_ID ] = NA
    # write.csv(oscc_sw, "output.csv")

    oscc_avg = aggregate(cur_run ~ pre_run + ID + group, data = subset(oscc_sw, pre_run < 11), FUN=median)

    oscc_avg$group = factor(oscc_avg$group, levels = rev(levels(oscc_avg$group)))
    ggplot(subset(oscc_avg, TRUE), aes(y=cur_run, x=as.factor(pre_run))) +
        stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(),  color="black", fill=brewer.pal(4, "OrRd")[2], size=0.2) +
        stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
                   position=position_dodge(.9),  fun.args = list(mult = 1)) +
        # geom_point(color="red", position=position_nudge(x = -0.3, y = 0), alpha=0.3, size=0.2) +
        geom_abline(intercept = 0, slope = 1, linetype='dashed')     +
        xlab("length of previous run") +
        ylab("length of current run") +
        scale_fill_brewer(name = "", palette=palette_mode) +
        blk_theme_no_grid(legend_position = "right", margins = c(2,2,2,2)) +
        # theme(panel.grid.minor.y = element_line(colour = text_color)) +
        facet_grid(.~group)

    ggsave(paste0("../doc/graphs/plots/", output_prefix, "_run_length_pre_cur.pdf"),
           width=5, height=2, units = "in", device=cairo_pdf)


    oscc_sw_percent = oscc_sw[, c("ID", "cur_run")]
    tbl = table(subset(oscc_sw_percent, TRUE))
    run_prop = prop.table(tbl, 1)
    require(reshape2)
    run_prop = melt(run_prop)
    diags = oscc_sw[!duplicated(oscc_sw$ID), c("ID", "group")]
    run_prop = merge(run_prop, diags, by = "ID")
    run_prop$group = factor(run_prop$group, levels = rev(levels(run_prop$group)))


    ggplot(subset(run_prop, cur_run < 15), aes(y=100 * value, x=as.factor(cur_run))) +
        stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(),  color="black", fill=brewer.pal(4, "OrRd")[2], size=0.2) +
        stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
                   position=position_dodge(.9),  fun.args = list(mult = 1)) +
        geom_point(color="red", position=position_nudge(x = -0.3, y = 0), alpha=0.3, size=0.2) +
        ylab("percentage of runs") +
        xlab("run length") +
        scale_fill_brewer(name = "", palette=palette_mode) +
        blk_theme_no_grid(legend_position = "right", margins = c(2,2,2,2)) +
        # theme(panel.grid.minor.y = element_line(colour = text_color)) +
        facet_grid(.~group)

    ggsave(paste0("../doc/graphs/plots/", output_prefix, "_run_percentage.pdf"),
           width=5, height=3, units = "in", device=cairo_pdf)

}
aggregate(value ~ diag, data = subset(run_prop, cur_run == 1), FUN = mean)

## stat analysis for the effect of reward

## effect of previous rewards
rew_key = read.csv("../nongit/results/archive/beh/to_graph_data/subj_all_data.csv")
rew_key = subset(rew_key, reward == 1)
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
rew_key$pre_rew_group = NA
rew_key[rew_key$prev_rewards == 0,]$pre_rew_group = "0"
rew_key[rew_key$prev_rewards == 1,]$pre_rew_group = "1"
rew_key[rew_key$prev_rewards == 2,]$pre_rew_group = "2"
rew_key[rew_key$prev_rewards > 2,]$pre_rew_group = ">2"
rew_key$same = (rew_key$same  == "True")* 1
agg_rew_key = aggregate(same ~ pre_rew_group + ID + group, data = rew_key, FUN=mean)


library(lmerTest)
library(lme4)

rew_key_stat = subset(agg_rew_key, pre_rew_group %in% c(0, '>2'))
head(rew_key)

m = lmer(same ~ pre_rew_group  + (1 |ID), data = subset(rew_key_stat, group == 'Healthy'))
print(summary(m))

m = lmer(same ~ pre_rew_group  + (1 |ID), data = subset(rew_key_stat, group == 'Depression'))
print(summary(m))

m = lmer(same ~ pre_rew_group  + (1 |ID), data = subset(rew_key_stat, group == 'Bipolar'))
print(summary(m))

## effect of previous actions
key_count_sw = read.csv("../nongit/results/archive/beh/to_graph_data/subj_all_data.csv")
key_count_sw$group = factor(key_count_sw$group, levels = rev(levels(key_count_sw$group)))
key_count_sw$same  = (key_count_sw$same == "True") * 1
key_count_sw = aggregate(same ~ prev_key + ID+ group, data = subset(key_count_sw, prev_key < 15 & reward == 0 & prev_rewards == 0), FUN=mean)

library(lmerTest)
library(lme4)
m = lmer(same ~ prev_key + (1 | ID), data = subset(key_count_sw, group =='Healthy'))
print(summary(m))

# number of key presses before switching when one reward is earned
rew_key = read.csv("../nongit/results/archive/beh/to_graph_data/subj_all_data.csv")
rew_key = subset(rew_key, prev_rewards ==1 & same == "False" & reward == 0)
agg_rew_key = aggregate(prev_key ~ ID, data = rew_key, FUN=mean)
mean(subset(agg_rew_key, T)$prev_key) + 1

rew_key = read.csv("../nongit/results/archive/beh/to_graph_data/subj_all_data.csv")
rew_key = subset(rew_key, prev_rewards ==2 & reward == 1)
mean(rew_key$prev_key)
agg_rew_key = aggregate(prev_key ~ ID + group, data = rew_key, FUN=mean)
mean(subset(agg_rew_key, T)$prev_key) + 1

################# not used for the paper ##########################################
## stat analysis for the effect of number of actions
rew_key = read.csv("../nongit/results/local/to_graph_data/subj_stats_pre_rew_key_ID.csv")
rew_key = subset(rew_key, pre_rew_group == 0)
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
rew_key$prev_key_group = factor(rew_key$prev_key_group, levels = c('<10', '>=10'))

m = lmer(same ~ prev_key_group  + (1 |ID), data = subset(rew_key, group == 'Healthy'))
print(summary(m))

m = lmer(same ~ prev_key_group  + (1 |ID), data = subset(rew_key, group == 'Depression'))
print(summary(m))

m = lmer(same ~ prev_key_group  + (1 |ID), data = subset(rew_key, group == 'Bipolar'))
print(summary(m))


m = lmer(same ~ prev_key_group * group  + (1 |ID), data = subset(rew_key, group %in% c('Healthy', 'Bipolar')))
print(summary(m))

m = lmer(same ~ prev_key_group * group  + (1 |ID), data = subset(rew_key, group %in% c('Bipolar', 'Depression')))
print(summary(m))


key_count = read.csv("../nongit/results/local/to_graph_data/subj_all_data.csv")
key_count$same  = (key_count$same == "True") * 1
key_count$rr = NA
key_count$rr[key_count$prev_rewards == 0] = "0"
key_count$rr[key_count$prev_rewards == 1] = "1"
key_count$rr[key_count$prev_rewards == 2] = "2"
key_count$rr[key_count$prev_rewards > 2] = ">2"
key_count = aggregate(same ~ rr + ID+ group, data = subset(key_count, reward == 1), FUN=mean)
# ggplot(subset(key_count, prev_key < 20 & reward == 1 & prev_rewards == 0), aes(y=same, x=prev_key)) +
subset(key_count, ID == "s_006_date_20100411_causality_results.mat")
ggplot(subset(key_count, T), aes(y=same, x=rr)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(),  color="black", fill=col, size=0.2) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
    scale_fill_brewer(name = "", palette=palette_mode) +
    blk_theme_no_grid(legend_position = "right") +
    facet_grid(.~group)


key_count = read.csv("../nongit/results/local/to_graph_data/subj_stats_pre_key_count.csv")
key_count$group = factor(key_count$group, levels = rev(levels(key_count$group)))
ggplot(subset(key_count, prev_key < 20 & same == "True"), aes(y = same, x=prev_key)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
    geom_smooth() +
    scale_fill_brewer(name = "", palette=palette_mode) +
    blk_theme_no_grid(legend_position = "right") +
    facet_grid(.~group)

rew_key = read.csv("../nongit/results/local/to_graph_data/subj_run_length_ID.csv")
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
rew_key$prev_key_group = factor(rew_key$prev_key_group, levels = c('<10', '10-20', '>20'))
ggplot(rew_key, aes(y = prev_key_group.1, x=prev_key_group, fill=group)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
    scale_fill_brewer(name = "", palette=palette_mode) +
    blk_theme_no_grid(legend_position = "right")

rew_key = read.csv("../nongit/results/local/to_graph_data/subj_stats_pre_key_ID.csv")
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
rew_key$prev_key_group = factor(rew_key$prev_key_group, levels = c('<10', '10-20', '>20'))
ggplot(rew_key, aes(y = same, x=prev_key_group, fill=group)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
    scale_fill_brewer(name = "", palette=palette_mode) +
    blk_theme_no_grid(legend_position = "right")


rew_key = read.csv("../nongit/results/local/to_graph_data/subj_stats_pre_rew_key_ID.csv")
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
rew_key$prev_key_group = factor(rew_key$prev_key_group, levels = c('<10', '10-20', '>20'))
require(ggplot2)
ggplot(rew_key, aes(x = prev_key_group, y = same, fill=pre_rew_group)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2 ) +
      stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
    scale_fill_brewer(name = "", palette=palette_mode) +
    blk_theme_no_grid(legend_position = "right") +
    facet_grid(.~group)


### previously in the paper
prev_key_levels = c("<10", "10-20", ">=10")

rew_key = read.csv("../nongit/results/local/to_graph_data/subj_stats_pre_rew_key.csv")
rew_key$prev_key_group = factor(rew_key$prev_key_group, levels = prev_key_levels)
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
require(ggplot2)
ggplot(rew_key, aes(x = prev_key_group, y = pre_rew_group)) +
    geom_tile(aes(fill = 100 * same)) +
    ylab("# previous rewards")+
    xlab("# previous actions")+
    scale_fill_distiller(name = "%stay", palette="Spectral") +
    # scale_x_discrete(labels = c("<10", expression(phantom(x) >=10))) +
    blk_theme_no_grid(legend_position = "right") +
    theme(panel.grid.major.y = element_blank())+
    facet_grid(.~group)
ggsave("../doc/graphs/plots/pre_rewards_actions.pdf", width=5, height=4, device=cairo_pdf)



rew_key = read.csv("../nongit/results/local/to_graph_data/subj_stats_pre_rew_ID.csv")
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
require(ggplot2)
library(RColorBrewer)
col = brewer.pal(4, "OrRd")[4]
p1 = ggplot(rew_key, aes(y = same, x=pre_rew_group)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", fill=col, size=0.2) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
    scale_fill_brewer(name = "", palette=palette_mode) +
    xlab("#previous rewards") +
    ylab('%stay') +
    blk_theme_no_grid(legend_position = "right") +
    facet_grid(.~group)



rew_key = read.csv("../nongit/results/local/to_graph_data/subj_stats_pre_rew_key_ID.csv")
rew_key$group = factor(rew_key$group, levels = rev(levels(rew_key$group)))
rew_key$prev_key_group = factor(rew_key$prev_key_group, levels = prev_key_levels)
require(ggplot2)
p2 = ggplot(subset(rew_key, pre_rew_group == 0), aes(x = prev_key_group, y = same)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", fill=col, size=0.2 ) +
      stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1)) +
    xlab("#previous actions") +
    ylab('%stay') +
    # scale_x_discrete(labels = c("<10", expression(phantom(x) >=10))) +
    scale_fill_brewer(name = "", palette=palette_mode) +
    blk_theme_no_grid(legend_position = "right") +
    facet_grid(.~group)

cairo_pdf("../doc/graphs/plots/prev_rew_action_2.pdf", width=20, height=2, onefile = FALSE)
require(gridExtra)
grid.arrange(p1, p2,ncol=2,nrow=1, widths=c(3/5, 2/5))
dev.off()
