paper_mode=TRUE

source('R/helper.R')
require(plyr)

######## for reward graphs ##########################
draw_sims('../nongit/results/archive/beh/sims/reward/GQL/', '-GQL', T)
draw_sims('../nongit/results/archive/beh/sims/reward/RNN/', '-RNN')
draw_sims('../nongit/results/archive/beh/sims/reward/QL/', '-QL', T)
draw_sims('../nongit/results/archive/beh/sims/reward/QLP/', '-QLP', T)
draw_sims('../nongit/results/archive/beh/sims/reward/GQL1D/', '-GQL-1D', T)
draw_sims('../nongit/results/archive/beh/sims/reward/LIN/', '-LIN', T)
draw_sims_multiple('../nongit/results/archive/beh/sims/reward/', '-RNN-mean', F)

draw_sims_osci('../nongit/results/archive/beh/sims/osci/', '-all-models')
draw_sims_healthy('../nongit/results/archive/beh/sims/reward/', '-all-models-reward')

draw_sims = function(input_path, file_name, policy_inlog=F){

    if (policy_inlog){
        yp = exp
    }else{
        yp = function(x){x}
    }
    polcs = list()
    trains = list()

    index = 1

    for (conds in c('1R1L', '1R2L', '1R3L')){
        for (group in c('Healthy', 'Bipolar', 'Depression')){
            address = paste0(input_path, conds, '/', group, '/')

            policies = read.csv(paste0(address, 'policies-.csv'))
            train = read.csv(paste0(address, 'train.csv'))

            policies$group = group
            train$group = group

            policies$conds = conds
            train$conds = conds

            policies$trial = seq(1:nrow(policies))

            polcs[[index]] = policies
            trains[[index]] = train

            index = index + 1

        }
    }

    require(data.table)

    policies = as.data.frame(rbindlist(polcs))
    train = as.data.frame(rbindlist(trains))


    n_actions = (ncol(policies) - 5)
    actions = policies[, c("X0", "X1", "trial", "group", "conds")]
    actions$reward = train$reward
    actions[actions$reward == 0, "reward"] = NA
    actions$states = train$states
    actions$selected_action = as.factor(train$choices)
    actions = melt(actions, variable_name = "action", id=c("trial", "reward", "states", "selected_action", "conds", "group"))
    actions$policy = actions$value
    actions$value = NULL

    actions$group = factor(actions$group)
    actions$group = factor(actions$group, levels = rev(levels(actions$group)))

    library(RColorBrewer)
    col = brewer.pal(10, "OrRd")


    require(ggplot2)
    ggplot(actions, aes(x = trial, y = yp(policy), fill=action, group=action)) +
        geom_area() + geom_line(position="stack")+
        geom_hline(yintercept=0.5, linetype="dashed",
                   color = "black", size=0.4)+
        scale_y_continuous(breaks=c(0.0, 0.5, 1.0)) +
        geom_point(aes(x = trial, y = reward + 0.1), color="black", fill="black", shape=4, size=1) +
        geom_point(aes(x = trial, y = -.2, color=selected_action, fill=selected_action), shape=22, size=2) +
        scale_fill_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[2], col[5])) +
        scale_color_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[2], col[5])) +
        guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
        ylab('probability of selecting L') +
        blk_theme_no_grid_nostrip(legend_position = "right", margins = c(1,1,1,1)) +
        geom_segment(aes(x=27, xend=27, y=1.6, yend=1.2),
                           arrow = arrow(length = unit(0.2, "cm"), type = "closed"), color = "red") +
        facet_grid(group~conds, labeller = labeller(conds = function(x){ rep("", length(x))}))


    ggsave(paste0("../doc/graphs/plots/sims_BD", file_name ,".pdf"),
           width=5, height=3, units = "in", device=cairo_pdf)

}

draw_sims_multiple = function(input_path, file_name, policy_inlog=F){

    if (policy_inlog){
        yp = exp
    }else{
        yp = function(x){x}
    }
    polcs = list()
    trains = list()

    index = 1

    for (run in c(0:14)){
        for (conds in c('1R1L', '1R2L', '1R3L')){
            for (group in c('Healthy', 'Bipolar', 'Depression')){
                address = paste0(input_path, 'RNN_', run, '/', conds, '/', group, '/')

                policies = read.csv(paste0(address, 'policies-.csv'))
                train = read.csv(paste0(address, 'train.csv'))

                policies$group = group
                train$group = group

                policies$conds = conds
                train$conds = conds

                policies$trial = seq(1:nrow(policies))

                polcs[[index]] = policies
                trains[[index]] = train

                index = index + 1

            }
        }
    }

    require(data.table)

    policies = as.data.frame(rbindlist(polcs))
    train = as.data.frame(rbindlist(trains))


    n_actions = (ncol(policies) - 5)
    actions = policies[, c("X0", "X1", "trial", "group", "conds")]
    actions$reward = train$reward
    actions[actions$reward == 0, "reward"] = NA
    actions$states = train$states
    actions$selected_action = as.factor(train$choices)
    actions = melt(actions, variable_name = "action", id=c("trial", "reward", "states", "selected_action", "conds", "group"))
    actions$policy = actions$value
    actions$value = NULL

    actions$group = factor(actions$group)
    actions$group = factor(actions$group, levels = rev(levels(actions$group)))

    library(RColorBrewer)
    col = brewer.pal(10, "OrRd")

    actions_mean = ddply(actions, .(trial, action, reward ,conds , group , selected_action), function(x){data.frame(policy = mean(yp(x$policy)), policy_std = sd(yp(x$policy)))})

    require(ggplot2)
    ggplot(actions_mean, aes(x = trial, y = policy, fill=action, group=action)) +
        geom_area() + geom_line(position="stack")+
        geom_hline(yintercept=0.5, linetype="dashed",
                   color = "black", size=0.4)+
        geom_ribbon(data=subset(actions_mean, action == "X1"), aes(x=trial, ymin=(policy - policy_std),ymax=(policy + policy_std)),alpha=0.5, fill = "black") +
        scale_y_continuous(breaks=c(0.0, 0.5, 1.0)) +
        geom_point(aes(x = trial, y = reward + 0.1), color="black", fill="black", shape=4, size=1) +
        geom_point(aes(x = trial, y = -.2, color=selected_action, fill=selected_action), shape=22, size=2) +
        scale_fill_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[2], col[5])) +
        scale_color_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[2], col[5])) +
        guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
        ylab('probability of selecting L') +
        blk_theme_no_grid_nostrip(legend_position = "right", margins = c(1,1,1,1)) +
        geom_segment(aes(x=27, xend=27, y=1.6, yend=1.2),
                           arrow = arrow(length = unit(0.2, "cm"), type = "closed"), color = "red") +
        facet_grid(group~conds, labeller = labeller(conds = function(x){ rep("", length(x))}))


    ggsave(paste0("../doc/graphs/plots/sims_BD", file_name ,".pdf"),
           width=5, height=3, units = "in", device=cairo_pdf)

}


draw_sims_healthy = function(input_path, file_name){

    polcs = list()
    trains = list()

    index = 1

    for (conds in c('1R1L', '1R2L', '1R3L')){
        for (group in c('Healthy')){
            for (model in c('RNN', 'LIN', 'GQL', 'QLP', 'QL')){
                address = paste0(input_path, model, '/', conds, '/', group, '/')

                policies = read.csv(paste0(address, 'policies-.csv'))
                train = read.csv(paste0(address, 'train.csv'))

                policies$group = group
                train$group = group

                policies$model = model
                train$model = model

                policies$conds = conds
                train$conds = conds

                policies$trial = seq(1:nrow(policies))

                if (model != 'RNN'){
                    policies[,1:3] = exp(policies[,1:3])
                }


                polcs[[index]] = policies
                trains[[index]] = train

                index = index + 1
            }

        }
    }

    require(data.table)

    policies = as.data.frame(rbindlist(polcs))
    train = as.data.frame(rbindlist(trains))


    n_actions = (ncol(policies) - 5)
    actions = policies[, c("X0", "X1", "trial", "group", "conds" ,"model")]
    actions$reward = train$reward
    actions[actions$reward == 0, "reward"] = NA
    actions$states = train$states
    actions$selected_action = as.factor(train$choices)
    actions = melt(actions, variable_name = "action", id=c("trial", "reward", "states", "selected_action", "conds", "group","model"))
    actions$policy = actions$value
    actions$value = NULL

    actions$model = factor(actions$model, levels = c("RNN", "LIN", "GQL", "QLP", "QL"))


    library(RColorBrewer)
    col = brewer.pal(10, "OrRd")


    require(ggplot2)
    ggplot(actions, aes(x = trial, y = policy, fill=action, group=action)) +
        geom_area() + geom_line(position="stack")+
        geom_hline(yintercept=0.5, linetype="dashed",
                   color = "black", size=0.4)+
        scale_y_continuous(breaks=c(0.0, 0.5, 1.0)) +
        geom_point(aes(x = trial, y = reward + 0.1), color="black", fill="black", shape=4, size=1) +
        geom_point(aes(x = trial, y = -.2, color=selected_action, fill=selected_action), shape=22, size=2) +
        scale_fill_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[2], col[5])) +
        scale_color_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[2], col[5])) +
        guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
        ylab('probability of selecting L') +
        blk_theme_no_grid_nostrip(legend_position = "right", margins = c(1,1,1,1)) +
        geom_segment(aes(x=27, xend=27, y=1.6, yend=1.2),
                           arrow = arrow(length = unit(0.2, "cm"), type = "closed"), color = "red") +
        facet_grid(model~conds, labeller = labeller(conds = function(x){ rep("", length(x))}))


    ggsave(paste0("../doc/graphs/plots/sims_BD", file_name ,".pdf"),
           width=5, height=4.6, units = "in", device=cairo_pdf)

}




draw_sims_osci = function(input_path, file_name){

    polcs = list()
    trains = list()

    index = 1

    for (conds in c('RNN', 'LIN', 'GQL', 'QLP', 'QL')){
        for (group in c('Healthy', 'Bipolar', 'Depression')){
            address = paste0(input_path, conds, '/', group, '/')

            policies = read.csv(paste0(address, 'policies-.csv'))
            train = read.csv(paste0(address, 'train.csv'))

            policies$group = group
            train$group = group

            policies$conds = conds
            train$conds = conds

            policies$trial = seq(1:nrow(policies))

            if (conds != 'RNN'){
                policies[,1:3] = exp(policies[,1:3])
            }

            polcs[[index]] = policies
            trains[[index]] = train

            index = index + 1

        }
    }

    require(data.table)

    policies = as.data.frame(rbindlist(polcs))
    train = as.data.frame(rbindlist(trains))


    n_actions = (ncol(policies) - 5)
    actions = policies[, c("X0", "X1", "trial", "group", "conds")]
    actions$reward = train$reward
    actions[actions$reward == 0, "reward"] = NA
    actions$states = train$states
    actions$selected_action = as.factor(train$choices)
    actions = melt(actions, variable_name = "action", id=c("trial", "reward", "states", "selected_action", "conds", "group"))
    actions$policy = actions$value
    actions$value = NULL

    actions$group = factor(actions$group)
    actions$group = factor(actions$group, levels = rev(levels(actions$group)))

    require(ggplot2)

    library(RColorBrewer)
    col = brewer.pal(10, "OrRd")

    actions$conds = factor(actions$conds, levels = c("RNN", "LIN", "GQL", "QLP", "QL"))

        ggplot(actions, aes(x = trial, y = policy, fill=action, group=action)) +
            geom_area() + geom_line(position="stack")+
            geom_hline(yintercept=0.5, linetype="dashed",
                       color = "black", size=0.4)+
            scale_y_continuous(breaks=c(0.0, 0.5, 1.0)) +
            geom_point(aes(x = trial, y = reward), color="red", fill="red", shape=8, size=1) +
            geom_point(aes(x = trial, y = -.2, color=selected_action, fill=selected_action), shape=22, size=2) +
            scale_fill_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[3], col[5])) +
            scale_color_manual(name = "", breaks=c("X0", "X1"), labels = c("R", "L"), values = c(col[3], col[5], col[3], col[5])) +
            scale_x_continuous(breaks=c(0.0, 9, 20)) +
            guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
            ylab('probability of selecting L') +
            blk_theme_no_grid(legend_position = "right", margins = c(1,1,1,1)) +
            geom_segment(aes(x=9.7, xend=20, y=-0.05, yend=-0.05),
                               , color = "blue", size=2) +
            geom_segment(aes(x=1, xend=9.3, y=-0.05, yend=-0.05),
                               , color = "green3", size=2) +
            facet_grid(group~conds)


    ggsave(paste0("../doc/graphs/plots/sims_BD", file_name ,".pdf"),
           width=5.0, height=4, units = "in", device=cairo_pdf)

}



# p2 = ggplot(subset(actions, conds == '1R3L' & action == 'X1' & trial > 55 & trial < 65), aes(x = trial, y = policy, color=group, linetype=group)) +
#     geom_line(size=1)+
#     geom_point(aes(x = trial, y = reward), color="red", fill="red", shape=8, size=1) +
#       scale_color_manual(values = c(Healthy = cols[3], Depression = cols[4], Bipolar = cols[5])) +
#       scale_shape_manual(values = c(Healthy = NA, Depression = 16, Bipolar = 17)) +
#     guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
#     ylim(c(0.4, 1)) +
#     labs(color="") +
#     labs(linetype="") +
#     blk_theme_no_grid_nostrip(legend_position = "right")
#
# library(gridExtra)
# pdf("../doc/graphs/plots/policy_zoomed.pdf", width=4, height=4, onefile = FALSE, title= '')
# grid.arrange(p1, p2,ncol=1,nrow=2)
# dev.off()


########## another simulation #############################################
require(reshape)
paper_mode = TRUE
source('R/helper.R')

addresses = list()
index = 1
n_cells = 10
for (group in c('Healthy', 'Bipolar', 'Depression')){
   addresses[[index]] = data.frame(
                         policies = paste("../nongit/results/local/sims/BD/", group, "/", n_cells, "cells/policies-.csv", sep = ""),
                         train = paste("../nongit/results/local/sims/BD/", group, "/", n_cells, "cells/train.csv", sep = ""),
                         output = paste("../nongit/graphs/policies/BD/", n_cells, "cells",  group, ".pdf", sep = ""),
                         group = group,
                         n_cells = n_cells
                         )
   index = index + 1
}
require(data.table)
addresses = as.data.frame(rbindlist(addresses))

for (r in c(1:nrow(addresses))){
  policies = read.csv(as.character(addresses[r,]$policies))
  train = read.csv(as.character(addresses[r,]$train))
  draw_policy(policies, train, as.character(addresses[r,]$output))
}




######## for oscillation graphs - not used in the paper ####################
polcs = list()
trains = list()

index = 1

for (group in c('Healthy')){
    address = paste0('../nongit/results/local/sims/BD/', 'oscc', '/', group, '/', '10cells/')

    policies = read.csv(paste0(address, 'policies-.csv'))
    train = read.csv(paste0(address, 'train.csv'))

    policies$group = group
    train$group = group

    policies$trial = seq(1:nrow(policies))

    polcs[[index]] = policies
    trains[[index]] = train

    index = index + 1

}

require(data.table)

policies = as.data.frame(rbindlist(polcs))
train = as.data.frame(rbindlist(trains))


n_actions = (ncol(policies) - 5)
actions = policies[, c("X0", "X1", "trial", "group")]
actions$reward = train$reward
actions[actions$reward == 0, "reward"] = NA
actions$states = train$states
actions$selected_action = train$choices
actions = melt(actions, variable_name = "action", id=c("trial", "reward", "states", "selected_action", "group"))
actions$policy = actions$value
actions$value = NULL

#actions$group = factor(actions$group)
#actions$group = factor(actions$group, levels = rev(levels(actions$group)))

require(ggplot2)
ggplot(actions, aes(x = trial, y = policy, fill=action, group=action)) +
    geom_area() + geom_line(position="stack")+
    geom_hline(yintercept=0.5, linetype="dashed",
               color = "black", size=0.4)+
    geom_point(aes(x = trial, y = reward), color="red", fill="red", shape=8, size=1) +
    scale_fill_brewer(name = "", palette=palette_mode, labels = c("A1", "A2")) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
    blk_theme_no_grid_nostrip(legend_position = "right") +
    facet_grid(group~.)


ggsave("../doc/graphs/plots/sims_BD.pdf",
       width=5.2, height=3, units = "in", device=cairo_pdf)
