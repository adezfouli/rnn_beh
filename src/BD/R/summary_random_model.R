

data =read.csv('../nongit/results/archive/beh/gql-ml-rand-opt/gql-ml-rand/accu.csv')

require(plyr)

ddply(data, "group", function(x){sum(x$total.nlp)})

data =read.csv('../nongit/results/archive/beh/ql-ml-rand-opt/ql-ml-rand/accu.csv')
ddply(data, "group", function(x){sum(x$total.nlp)})

data =read.csv('../nongit/results/archive/beh/qlp-ml-rand-opt/qlp-ml-rand/accu.csv')
ddply(data, "group", function(x){sum(x$total.nlp)})