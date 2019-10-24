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

require(randtests)

ids = unique(r_data$ID)
rnd_test = matrix(ncol=2, nrow=101)
r = 1
for (id in ids){

    seq_keys = subset(r_data, ID == id)$key
    rnd_test[r, 2] = runs.test(seq_keys, threshold=0.5)$p.value
    rnd_test[r, 1] = id
    r = r + 1
}


rnd_test = data.frame(ID = rnd_test[, 1], p = as.numeric(rnd_test[, 2]))

diags = r_data[!duplicated(r_data$ID), c("group", "ID")]

rnd_test = merge(diags, rnd_test, by = "ID")

subset(rnd_test, p > 0.001)
