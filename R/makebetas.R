## Simulate Betas and covariates

# 25 realizations
#1. 2 covs, log(beta) = beta_0 + beta_1*1(high density)*(high wat san) beta_1 = log(2), beta_0 = log(1.5)
#2. log(beta) = beta_0 + beta_1*V1 + ... + beta_100*V100 (beta is uniform from -0.5 to 0.5)

gen_betas_effect_mod <- function(n) {
##scenario 1: interaction
#set.seed(2213)

    hd <- rbinom(n, 1, 0.5) #indicator of high pop density
    ws <- rbinom(n, 1, 0.5) # indicator of good wash indicators
    #beta1 <- exp(log(runif(n, min = 1, max = 2)) + log(2) * (hd == 1) * (ws == 0))
    beta1 <- exp(log(1.5) + log(2) * (hd == 1) * (ws == 0))
    #ds1 <- as.data.frame(cbind(hd, ws, beta))
#check
    #aggregate(beta1 ~ unlist(hd) + unlist(ws), data = ds1,  mean)


    return(data.frame(beta1=beta1, hd=hd, ws=ws))
}

gen_betas_many_cov <- function(n, num.betas) {
    ##scenario 2: beta depends on many correlated vars
    v <- matrix(nrow = n, ncol = num.betas) #100 predictor vars
    colnames(v) <-paste("V",1:num.betas,sep="")
    gamma <- (rnorm(n, mean = .693, sd = 0.25)) #correlation factor used blow
    for (i in 1:ncol(v)){
        if(i <= round(num.betas/10)) {v[,i] <- runif(n, min = 1, max = 2)} #1 through 10 are higher than the others
        if(round(num.betas/10) < i & i<= round(9*num.betas/10)) {v[,i] <- runif(n, min = -0.5, max = 0.5)} #10 - 90 are independent
        if(i > 9*round(num.betas/10)) {v[,i] <- gamma * v[,(i -  round(8*num.betas/10))]} #91 - 100 are very dependent on 11 - 20
    }
    alpha <- runif(num.betas, min = -.5, max = .5)
    beta2 <- 1 + as.vector(exp(log(1.5) + ((alpha) %*% t(v)) - sum(alpha * apply(v, 2, mean))))
    return(as.data.frame(cbind(beta2,v)))
}

## Write csv files ------
# setwd("/Users/jkedwar/Box Sync/Research/epidemics/R/")
# sc1 <- cbind(beta1, hd, ws)
# write.csv(sc1, file = "sc1.csv")
#
# sc2 <- cbind(beta2, v)
                                        # write.csv(sc2, file = "sc2.csv")

gen_betas_many_cov2 <- function(n, num_pred, num_not_pred) {


    ##all variables in range -1, 1 uniformly
    v <- matrix(runif(n*(num_pred+num_not_pred),-1,1) ,nrow=n, ncol=num_pred + num_not_pred)
    alpha <- c(rep(1, num_pred),rep(0,num_not_pred))

    colnames(v) <-paste("V",1:(num_pred + num_not_pred),sep="")
    beta2 <- 1 + as.vector(exp(alpha%*%t(v)))
    return(as.data.frame(cbind(beta2, v)))

}
