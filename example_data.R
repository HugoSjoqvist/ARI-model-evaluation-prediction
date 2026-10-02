############################################################
# This code simulates a dataset of 20,000 individuals. 
# The data is not supposed to have a logical interpretation,
# but is only used for a practical purpose. 
#
# Note that the sex variable is not wanted in the ARI score
# (only created as an example of how the code can exclude it). 
# 
# Here we simulate the binary outcome "bip", and use information
# from the siblings and cousins - both number of that specific
# family member, and the number of that type who have "bip" or "adhd". 
############################################################

set.seed(12345)
n <- 20000
# ID variable
id <- 1:n
# Birth year (uniform 1992-2004)
birthyear <- sample(1992:2004, n, replace = TRUE)
# Six variables between 0-15,
# strongly skewed toward lower values
simulate_var <- function(n) {
  rbeta(n, shape1 = 1.2, shape2 = 5) * 15
}
n_siblings <- round(simulate_var(n), 0)
n_asd_siblings <- round(simulate_var(n), 0)
n_bip_siblings <- round(simulate_var(n), 0)
n_cousins <- round(simulate_var(n), 0)
n_bip_cousins <- round(simulate_var(n), 0)
n_asd_cousins <- round(simulate_var(n), 0)
# Binary outcomes
bip <- rbinom(n, size = 1, prob = 0.10)
sex <- rbinom(n, size = 1, prob = 0.50)
n_bip_parents <- rbinom(n, size = 2, prob = 0.50)
n_asd_parents <- rbinom(n, size = 2, prob = 0.20)
# Final dataset
example_bip <- data.frame(
  id,
  birthyear,
  bip,
  sex,
  n_bip_parents,
  n_asd_parents,
  n_siblings,
  n_asd_siblings,
  n_bip_siblings,
  n_cousins,
  n_bip_cousins,
  n_asd_cousins
)


