library(data.table)
library(digest)

require(data.table)
r_data = data.table(read.csv(unz(description = "../data/BD/choices_diagno.csv.zip", filename = 'choices_diagno.csv')))


anonymize <- function(x, algo="crc32"){
  unq_hashes <- vapply(unique(x), function(object) digest(object, algo=algo), FUN.VALUE="", USE.NAMES=TRUE)
  unname(unq_hashes[x])
}

cols_to_mask <- c("ID")

r_data[,cols_to_mask := lapply(.SD, anonymize),.SDcols=cols_to_mask,with=FALSE]

r_data$block = r_data$trial
r_data$outcome = NULL
r_data$trial = NULL
r_data$condition = NULL
r_data$code = NULL
r_data$PressRate.sec = NULL
r_data$Mania = NULL
r_data$Depression = NULL
r_data$time = NULL
r_data$X = NULL

r_data$best_action = as.character(r_data$choice)
r_data$choice = NULL

r_data[r_data$best_action == 'R1', ]$best_action = "TRUE"
r_data[r_data$best_action == 'R2', ]$best_action = "FALSE"

write.csv(r_data, '../data/BD/for_plos.csv', row.names = FALSE)

################ data check ###################

org_data = read.csv(unz(description = "../data/BD/choices_diagno.csv.zip", filename = 'choices_diagno.csv'))
plos_data =read.csv(unz(description = "../data/BD/for_plos.csv.zip", filename = 'for_plos.csv'))

length(unique(org_data$ID))
length(unique(plos_data$ID))

for (i in c(1:101)){
  org_d = subset(org_data, ID == unique(org_data$ID)[i])
  plos_d = subset(plos_data, ID == unique(plos_data$ID)[i])

  print(nrow(org_d) == nrow(plos_d))
  print(all(org_d[,c("key", "trial", "reward", "diag")] == plos_d[,c("key", "block", "reward", "diag")]) )
}


library(lme4)
m = glmer(best_action ~ 1 + (1| ID), data = subset(plos_data, diag == 'Depression'), family = binomial())

print(summary(m))
