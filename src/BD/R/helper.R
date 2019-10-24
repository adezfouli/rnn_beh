text_color = "white"
back_color = "black"
error_bar_colour = "white"
palette_mode = "Set1"
f_size = 20

if (exists("paper_mode")){
  if(paper_mode){
    text_color = "black"
    back_color = "white"
    error_bar_colour = "black"
    palette_mode = "OrRd"
    f_size = 10
  }
}


blk_theme = function(fsize=f_size, rotate_x = FALSE, legend_position="right", margins = c(0,0,0,0)){

  if(rotate_x){
    xt = element_text(angle=45, hjust=1)
  }else{
    xt = element_text()
  }

  theme(legend.direction = "vertical", legend.position = legend_position, legend.box = "vertical",
        axis.text = element_text(colour = text_color),
        legend.text = element_text(colour = text_color),
        axis.line = element_line(colour = text_color),
        panel.border = element_blank(),
        text=element_text(size=fsize, colour = text_color),
        legend.key = element_blank(),
        axis.title.x = element_text(colour = text_color),
        axis.title.y = element_text(colour = text_color),
        axis.text.x=xt,
        panel.background = element_rect(fill = back_color),
        plot.background = element_rect(fill = back_color, colour=NA),
        legend.background = element_rect(fill = back_color),
        plot.margin=unit(margins,"mm"),
        panel.grid.major.y = element_line(colour = text_color)
  )
}


blk_theme_no_grid = function(fsize=f_size, rotate_x = FALSE, legend_position="right", margins = c(0,0,0,0)){

  if(rotate_x){
    xt = element_text(angle=45, hjust=1)
  }else{
    xt = element_text()
  }

  theme(legend.direction = "vertical", legend.position = legend_position, legend.box = "vertical",
        axis.text = element_text(colour = text_color),
        legend.text = element_text(colour = text_color),
        axis.line = element_line(colour = text_color),
        panel.border = element_blank(),
        text=element_text(size=fsize, colour = text_color),
        legend.key = element_blank(),
        axis.title.x = element_text(colour = text_color),
        axis.title.y = element_text(colour = text_color),
        axis.text.x=xt,
        panel.background = element_rect(fill = back_color),
        plot.background = element_rect(fill = back_color, colour=NA),
        legend.background = element_rect(fill = back_color),
        plot.margin=unit(margins,"mm"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(colour = text_color)
  )
}


blk_theme_no_grid_nostrip = function(fsize=f_size, rotate_x = FALSE, legend_position="right", margins = c(0,0,0,0)){

  if(rotate_x){
    xt = element_text(angle=45, hjust=1)
  }else{
    xt = element_text()
  }

  theme(legend.direction = "vertical", legend.position = legend_position, legend.box = "vertical",
        axis.text = element_text(colour = text_color),
        legend.text = element_text(colour = text_color),
        axis.line = element_line(colour = text_color),
        panel.border = element_blank(),
        text=element_text(size=fsize, colour = text_color),
        legend.key = element_blank(),
        axis.title.x = element_text(colour = text_color),
        axis.title.y = element_text(colour = text_color),
        axis.text.x=xt,
        panel.background = element_rect(fill = back_color),
        plot.background = element_rect(fill = back_color, colour=NA),
        legend.background = element_rect(fill = back_color),
        plot.margin=unit(margins,"mm"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(colour = text_color),
        strip.text.x = element_blank()
  )
}


blk_theme_nothing = function(fsize=f_size, rotate_x = FALSE, legend_position="right", margins = c(0,0,0,0)){

  if(rotate_x){
    xt = element_text(angle=45, hjust=1)
  }else{
    xt = element_text()
  }

  theme(legend.direction = "vertical", legend.position = legend_position, legend.box = "vertical",
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.text = element_blank(),
        axis.line = element_line(colour = text_color),
        panel.border = element_blank(),
        text=element_blank(),,
        legend.key = element_blank(),
        axis.title = element_blank(),
        plot.background = element_rect(fill = back_color, colour=NA),
        legend.background = element_rect(fill = back_color),
        plot.margin=unit(margins,"mm"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        strip.text.y = element_blank(),
        panel.spacing.x = unit(0, "lines")
  )
}


blk_theme_grid_hor = function(fsize=f_size, rotate_x = FALSE, legend_position="right", margins = c(0,0,0,0)){

  if(rotate_x){
    xt = element_text(angle=45, hjust=1)
  }else{
    xt = element_text()
  }

  theme(legend.direction = "vertical", legend.position = legend_position, legend.box = "vertical",
        axis.text = element_text(colour = text_color),
        legend.text = element_text(colour = text_color),
        axis.line = element_line(colour = text_color),
        panel.border = element_blank(),
        text=element_text(size=fsize, colour = text_color),
        legend.key = element_blank(),
        axis.title.x = element_text(colour = text_color),
        axis.title.y = element_text(colour = text_color),
        axis.text.x=xt,
        panel.background = element_rect(fill = back_color),
        plot.background = element_rect(fill = back_color, colour=NA),
        legend.background = element_rect(fill = back_color),
        plot.margin=unit(margins,"mm"),
        panel.grid.major.x  = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour = text_color)
  )
}



blk_theme_grid_hor_nox = function(fsize=f_size, legend_position="right", margins = c(0,0,0,0), legend.direction = "vertical"){


  theme(legend.direction = legend.direction, legend.position = legend_position, legend.box = "horizontal",
        axis.text = element_text(colour = text_color),
        legend.text = element_text(colour = text_color),
        axis.line = element_line(colour = text_color),
        panel.border = element_blank(),
        text=element_text(size=fsize, colour = text_color),
        legend.key = element_blank(),
        axis.title.y = element_text(colour = text_color),
        panel.background = element_rect(fill = back_color),
        plot.background = element_rect(fill = back_color, colour=NA),
        legend.background = element_rect(fill = back_color),
        plot.margin=unit(margins,"mm"),
        panel.grid.major.x  = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour = text_color),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()
  )
}


blk_theme_no_lablel = function(fsize=f_size, rotate_x = FALSE, legend_position="right", margins = c(0,0,0,0)){

  if(rotate_x){
    xt = element_text(angle=45, hjust=1)
  }else{
    xt = element_text()
  }

  theme(legend.direction = "vertical", legend.position = legend_position, legend.box = "vertical",
        legend.text = element_text(colour = text_color),
        axis.line = element_line(colour = text_color),
        panel.border = element_blank(),
        text=element_text(size=fsize, colour = text_color),
        legend.key = element_blank(),
        axis.title = element_blank(),
        axis.text=element_blank(),
        axis.ticks=element_blank(),
        panel.background = element_rect(fill = back_color),
        plot.background = element_rect(fill = back_color, colour=NA),
        legend.background = element_rect(fill = back_color),
        plot.margin=unit(margins,"mm"),
        panel.grid.major.x  = element_blank(),
        panel.grid.minor.x = element_blank(),
        strip.background = element_blank(),
        strip.text.x = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(colour = text_color)

  )
}


require(reshape)
draw_policy = function(policies, train, output){
  n_actions = (ncol(policies) - 3)
  actions = policies[, 2:(1+n_actions)]
  actions$trial = seq(1:nrow(actions))
  actions$reward = train$reward
  actions[actions$reward == 0, "reward"] = NA
  actions$states = train$states
  actions$selected_action = train$choices
  actions = melt(actions, variable_name = "action", id=c("trial", "reward", "states", "selected_action"))
  actions$policy = actions$value
  actions$value = NULL

  require(ggplot2)
  ggplot(actions, aes(x = trial, y = policy, fill=action, group=action)) +
    geom_area() + geom_line(position="stack")+
    geom_hline(yintercept=0.5, linetype="solid",
               color = back_color, size=1)+
    geom_point(aes(x = trial, y = reward), color="cyan", fill="cyan", shape=8, size=3) +
    geom_point(aes(x = trial, y = 0, shape = as.factor(selected_action)), size=3, color="cyan", fill="cyan") +
    scale_fill_brewer(name = "", palette="Set1", labels = c("X0", "X1")) +
    scale_shape_manual(name = "",
                       labels = c("X0", "X1"),
                       values = c("0", "1")) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0)) +
    blk_theme_no_grid(legend_position = "right")
    ggsave(output,  width=30, height=8, units = "cm")
}
