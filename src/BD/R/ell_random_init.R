data =read.csv('../nongit/results/archive/beh/rnn-opt-rand-init/accu.csv')

require(plyr)

ddply(data, "group", function(x){data.frame(mean  = mean(x$total.nlp), sd = sd(x$total.nlp))})

