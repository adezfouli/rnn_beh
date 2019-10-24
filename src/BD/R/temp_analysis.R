sum_best = 0
sum_nonbset = 0
stay_rew = 0
stay_nrew = 0

sw_rew = 0
sw_nrew = 0

dname = 'sim_0DDJOXB'

dlist = list.dirs(path = '../nongit/results/archive/beh/BD_QRL/sims/', full.names = FALSE, recursive = FALSE)

for (dname in dlist){

    dir_in = paste0('../nongit/results/archive/beh/BD_QRL/sims/', dname, '/')
    conf = read.csv(paste0(dir_in, 'config.csv'))

    id = conf$ID

    if (id == 's_059_date_20072011_causality_results.mat' & conf$option == 'persv_False-'){
    train = read.csv(paste0(dir_in, 'train.csv'))

    if (conf$prop0 > conf$prop1){
        best_a = 0
    }else{
        best_a = 1
    }

    s_train = train$choice[1:(nrow(train) - 1)]
    s_train2 = train$choice[2:(nrow(train) )]
    r = train$reward[1:(nrow(train) - 1)]

    stay_rew = stay_rew + sum(s_train[r == 1] == s_train2[r == 1])
    stay_nrew = stay_nrew + sum(s_train[r == 0] == s_train2[r == 0])

    sw_rew = sw_rew + sum(s_train[r == 1] != s_train2[r == 1])
    sw_nrew = sw_nrew + sum(s_train[r == 0] != s_train2[r == 0])

    sum_best = sum_best + sum(train$choice == best_a)
    sum_nonbset = sum_nonbset + sum(train$choice != best_a)
    }
}

# temp test for number of actions and rewards
require(plyr)
r_data = read.csv(unz(description = "../data/BD/choices_diagno.csv.zip", filename = 'choices_diagno.csv'))
r_data$reward = r_data$outcome != "null"
all_data = list()
index = 1

for (id in unique(r_data$ID)[1:5]){
    for (t in unique(r_data$trial)){
        s_data = subset(r_data, ID == id & trial == t)
        rl = rle(as.character(s_data$key))
        s_data$prev_actions = sequence(rl$lengths) - 1
        s_data <- transform(s_data, indexer = rep(1:length(rl$lengths), rl$lengths))
        s_data$pre_rewards = ddply(s_data, "indexer", summarize, V1 = cumsum(c(0, head(reward, -1))))$V1
        s_data$same = 1 * (as.character(s_data$key) == c(as.character(tail(s_data$key, -1)), NA))
        s_data[nrow(s_data), 'same'] = 0
        all_data[[index]] = s_data
        index = index + 1
    }
}

require(data.table)
all_data_2 = as.data.frame(rbindlist(all_data))

all_data_2$pre_rewards_group = ">2"
all_data_2[all_data_2$pre_rewards == 0, ]$pre_rewards_group = "0"
all_data_2[all_data_2$pre_rewards == 1, ]$pre_rewards_group = "1"
all_data_2[all_data_2$pre_rewards == 2, ]$pre_rewards_group = "2"

all_data_2$pre_action_group = ">=10"
all_data_2[all_data_2$prev_actions  < 10, ]$pre_action_group = "<10"

ddply(subset(all_data_2, reward), c("ID", "pre_rewards_group"), function(x){data.frame(value = mean(x$same, na.rm= T))})
ddply(subset(all_data_2, reward), c("ID", "pre_action_group"), function(x){data.frame(value = mean(x$same, na.rm= T))})

ddply(subset(all_data_2, reward), c("pre_rewards_group", "pre_action_group", "diag"), function(x){data.frame(value = mean(x$same, na.rm= T))})

unique(all_data$ID)

all_data_2[, c('reward', 'key', 'pre_rewards', 'prev_actions', "pre_rewards_group")]
