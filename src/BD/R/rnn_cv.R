require(ggplot2)

#set to false to get presentation style
paper_mode= TRUE
source('R/helper.R')

library(RColorBrewer)
graph_data = read.csv("../nongit/results/archive/beh/rnn-cv/rnn-cv-evals/accu.csv")
graph_data$iter = substr(graph_data$model_iter, start = 7, stop=10)

graph_data = subset(graph_data, iter != 'fina')
graph_data$iter = as.numeric(graph_data$iter)

graph_data$group = factor(graph_data$group, levels = rev(levels(graph_data$group)))


col = brewer.pal(4, "OrRd")[4]
#plotting correct predictions
p1 = ggplot(subset(graph_data, TRUE), aes(x = iter, y = 100 * acc)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
  xlab("#training iterations") +
  ylab('%correct') +
  ylim(c(0, 100)) + 
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = T) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) + 
  facet_grid(group~cell, labeller = labeller(cell = function(x){paste0(x, ' cells')}))


#plotting NLP
p2 = ggplot(subset(graph_data, TRUE), aes(x = iter, y = nlp)) +
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
  stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
               position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
  xlab("#training iterations") +
  ylab('NLP') +
  scale_fill_brewer(name = "", palette=palette_mode) +
  blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = T) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
  facet_grid(group~cell, labeller = labeller(cell = function(x){paste0(x, ' cells')}))


require(gridExtra)
cairo_pdf("../doc/graphs/plots/RNN_CV_cells.pdf", width=5.2, height=7.5, onefile = FALSE)
grid.arrange(p1, p2,ncol=1,nrow=2)
dev.off()

# best model averaged all the subjects
avg_acc = aggregate(nlp ~ cell + model_iter, data = graph_data, mean)
avg_acc[which.min(avg_acc$nlp),]

avg_acc = aggregate(nlp ~ cell + model_iter, data = subset(graph_data, group == 'Bipolar'), mean)
avg_acc[which.min(avg_acc$nlp),]

avg_acc = aggregate(nlp ~ cell + model_iter, data = subset(graph_data, group == 'Depression'), mean)
avg_acc[which.min(avg_acc$nlp),]

avg_acc = aggregate(nlp ~ cell + model_iter, data = subset(graph_data, group == 'Healthy'), mean)
avg_acc[which.min(avg_acc$nlp),]


# for finding best hyper parameters among other groups
print('healthy')
avg_acc = aggregate(nlp ~ cell + model_iter, data = subset(graph_data, group %in% c('Bipolar', 'Depression')), mean)
opt = avg_acc[which.min(avg_acc$nlp),]

mean(subset(graph_data, group == 'Healthy' & cell == opt$cell & model_iter == opt$model_iter)$nlp)

print('depression')
avg_acc = aggregate(nlp ~ cell + model_iter, data = subset(graph_data, group %in% c('Bipolar', 'Healthy')), mean)
opt=avg_acc[which.min(avg_acc$nlp),]

mean(subset(graph_data, group == 'Depression' & cell == opt$cell & model_iter == opt$model_iter)$nlp)

print('bipolar')
avg_acc = aggregate(nlp ~ cell + model_iter, data = subset(graph_data, group %in% c('Healthy', 'Depression')), mean)
opt=avg_acc[which.min(avg_acc$nlp),]

mean(subset(graph_data, group == 'Bipolar' & cell == opt$cell & model_iter == opt$model_iter)$nlp)
