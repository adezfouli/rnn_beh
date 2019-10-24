d_total = read.csv(unz(description = "../data/BD/choices_diagno.csv.zip", filename = 'choices_diagno.csv'))

# subjects in each gropu
require(plyr)
tmp1 = ddply(d_total, c("diag"), function(x){data.frame(length(unique(x$ID)))})


# number of choices in each group
require(plyr)
tmp1 = ddply(d_total, c("ID", "diag", "trial"), function(x){nrow(x)})
aggregate(V1 ~ diag, mean, data=tmp1)

# subjects in each group
ddply(unique(d_total[,c("ID", "diag")]), "diag", function(x){nrow(x)} )




require(plyr)
tmp1 = ddply(d_total, c ("ID", "diag", "trial"), function(x){nrow(x)})
aa = subset(aggregate(V1 ~ diag + trial, mean, data=tmp1), diag == "Depression")
aa


paste(as.integer(aa$V1), collapse = ',')
