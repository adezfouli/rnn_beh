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

init_data= data.frame(key = 0.5, reward= 0)
init_data = rbind(
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data,
            init_data
)

to_model_data = function(d){
    data.frame(
                        reward1 = d[1, 'reward'],
                        reward2 = d[2, 'reward'],
                        reward3 = d[3, 'reward'],
                        reward4 = d[4, 'reward'],
                        reward5 = d[5, 'reward'],
                        reward6 = d[6, 'reward'],
                        reward7 = d[7, 'reward'],
                        reward8 = d[8, 'reward'],
                        reward9 = d[9, 'reward'],
                        reward10 = d[10, 'reward'],
                        reward11 = d[11, 'reward'],
                        reward12 = d[12, 'reward'],
                        reward13 = d[13, 'reward'],
                        reward14 = d[14, 'reward'],
                        reward15 = d[15, 'reward'],
                        reward16 = d[16, 'reward'],
                        reward17 = d[17, 'reward'],
                        reward18 = d[18, 'reward'],
                        key1 = d[1, 'key'],
                        key2 = d[2, 'key'],
                        key3 = d[3, 'key'],
                        key4 = d[4, 'key'],
                        key5 = d[5, 'key'],
                        key6 = d[6, 'key'],
                        key7 = d[7, 'key'],
                        key8 = d[8, 'key'],
                        key9 = d[9, 'key'],
                        key10 = d[10, 'key'],
                        key11 = d[11, 'key'],
                        key12 = d[12, 'key'],
                        key13 = d[13, 'key'],
                        key14 = d[14, 'key'],
                        key15 = d[15, 'key'],
                        key16 = d[16, 'key'],
                        key17 = d[17, 'key'],
                        key18 = d[18, 'key'])
}


simulate_model = function(model, T, get_reward_action, init_data){

    model_data= init_data
    sim_train = data.frame(reward = rep(-1, T), choices = rep(-1, T), pol = rep(-1, T))
    for (t in c(1:T)){
        pred = predict(model, to_model_data(model_data), type= "response")
        sim_train[t, 'pol'] = pred

        ra = get_reward_action(t, pred)
        model_data = rbind(data.frame(key = ra$action, reward = ra$reward ), model_data[1:(nrow(model_data) -1 ), ])
        sim_train[t, 'reward'] = ra$reward
        sim_train[t, 'choices'] = ra$action
    }
    sim_train
}

get_reward_action = function(trial, pol){
    data.frame(reward = 0, action = 0)
}

get_reward_action_from_file = function(input_csv){
    function(trial, pol){
        data.frame(reward = input_csv[trial, 'reward'], action = input_csv[trial, 'choices'])
    }
}

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

############ off-policy simulations ###############################
for (g in c('Healthy', 'Bipolar', 'Depression')){

        for (cond in c('1R1L', '1R2L', '1R3L')){
            train_file = read.csv(paste0('../nongit/results/archive/beh/sims/reward/GQL/', cond, '/', g, '/train.csv'))
            output = simulate_model(model_list[[g]], nrow(train_file), get_reward_action_from_file(train_file), init_data)
            output$states = NA
            output$id = 'id1'
            output$trial = 1
            pol = output
            pol = data.frame(X0 = log( 1 - output$pol), X1 = log( output$pol), id = 'id1', trial = 1)
            output$pol = NULL
            write.csv(output, paste0('../nongit/results/archive/beh/sims/reward/LIN/', cond, '/', g, '/train.csv'))
            write.csv(pol, paste0('../nongit/results/archive/beh/sims/reward/LIN/', cond, '/', g, '/policies-.csv'))
        }
}

############ off/on-policy simulations ###############################

on_off_policy = function(input_csv){
    function(trial, pol){
        if (trial < 10){
            data.frame(reward = input_csv[trial, 'reward'], action = input_csv[trial, 'choices'])
        }else{
            if (pol < 0.5){
                a = 0
            }else{
                a = 1
            }
            data.frame(reward = 0, action = a)
        }
    }
}


train_file = read.csv(paste0('../nongit/results/archive/beh/sims/osci/LIN/train.csv'))
for (g in c('Healthy', 'Bipolar', 'Depression')){

        output = simulate_model(model_list[[g]], 20, on_off_policy(train_file), init_data)
        output$states = NA
        output$id = 'id1'
        output$trial = 1
        pol = output
        pol = data.frame(X0 = log( 1 - output$pol), X1 = log( output$pol), id = 'id1', trial = 1)
        output$pol = NULL
        write.csv(output, paste0('../nongit/results/archive/beh/sims/osci/LIN/', g, '/train.csv'))
        write.csv(pol, paste0('../nongit/results/archive/beh/sims/osci/LIN/', g, '/policies-.csv'))
}


############### on sims ##############################################

gql_path = '../nongit/results/archive/beh/gql-ml-opt/gql-ml/'
gql_dirs = list.dirs(gql_path, full.names = TRUE)[-1]


on_policy = function(prop0, prop1){
    function(trial, pol){

            a = rbinom(1, 1, as.numeric(pol))
            r = (1 - a) * rbinom(1, 1, prop0) + a * rbinom(1, 1, prop1)
            data.frame(reward = r, action = a)
        }
}

for (conf_dir in gql_dirs){
    sim_conf = read.csv(paste0(conf_dir, '/config.csv'))
    output = simulate_model(model_list[[as.character(sim_conf$group)]], sim_conf$choices, on_policy(sim_conf$prop0, sim_conf$prop1), init_data)
    output$states = NA
    output$id = 'id1'
    output$trial = 1
    pol = output
    pol = data.frame(a0 = log( 1 - output$pol), a1 = log( output$pol), id = 'id1', trial = 1)
    output$pol = NULL
    new_dir = paste0('sims_', stringi::stri_rand_strings(1, 8))
    dir.create(paste0('../nongit/results/archive/beh/lin-on-sims/', new_dir))
    write.csv(output, paste0('../nongit/results/archive/beh/lin-on-sims/', new_dir , '/train.csv'))
    write.csv(pol, paste0('../nongit/results/archive/beh/lin-on-sims/', new_dir, '/policies-.csv'))
    write.csv(sim_conf, paste0('../nongit/results/archive/beh/lin-on-sims/', new_dir, '/config.csv'))
}

######### for testing ###################################################
######### for testing ###################################################
dd =   data.frame(
                        key1 = 0,
                        key2 = 0,
                        key3 = 0,
                        key4 = 0,
                        key5 = 0,
                        key6 = 0,
                        key7 = 0,
                        key8 = 0,
                        key9 = 0,
                        key10 = 0,
                        key11 = 0,
                        key12 = 0,
                        key13 = 0,
                        key14 = 0,
                        key15 = 0,
                        key16 = 0,
                        key17 = 0,
                        key18 = 0,
                        reward1 = 1,
                        reward2 = 0,
                        reward3 = 0,
                        reward4 = 0,
                        reward5 = 1,
                        reward6 = 0,
                        reward7 = 1,
                        reward8 = 1,
                        reward9 = 0,
                        reward10 = 0,
                        reward11 = 0,
                        reward12 = 0,
                        reward13 = 0,
                        reward14 = 0,
                        reward15 = 0,
                        reward16 = 0,
                        reward17 = 0,
                        reward18 = 0
)

    mm =  glm(key ~ 1 +
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
            data = subset(g_data, group == 'Depression'), family = binomial)


log(1 - predict(mm, dd, type = 'response'))
######################################################################

######################################################################
