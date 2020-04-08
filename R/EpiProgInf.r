##require(mgcv)

##'convience function for the logistic transform
##' @param x the number to be transformed
##'
##' @return the logistic transform


logistic <- function(x){
  return(log(x/(1-x)))
}

##'convience function for the inverse logistic transform
##' @param x the number to be transformed
##'
##' @return the inverse logistic transform

inv_logistic <- function(x) {
  return(exp(x)/(1+exp(x)))
}


##' Function to estimate the progress on an epidemic
##' using the method of Viboud and Chowell.
##'
##' @param epi.curve an epidemic curve
##' @param cum.sum the cumulative sums to use...allows us to pass
##'        in epi.curve with missing observations.
##' @param r intrinsic growth parameter
##' @param p "deceleration of growth' parameter
##' @param K final size of the outbreak
##' @param ... additional options to optim
##'
##' @return expected final size of outbreak and params specifying outbreak

final.size.ViboudChowell <- function(epi.curve,
                                     cum.curve = cumsum(epi.curve),
                                     r =1,
                                     p = 0.5,
                                     K = max(cum.curve),...){

  log.K <- log(K)
  log.r <- log(r)
  logistic.p <- logistic(p)

  ## NOTE: when using cum.curve=cumsum(epi.curve), cum.curve is using whatever epi.curve is when it is first used
  cum.curve <- cum.curve[-length(cum.curve)] ##this was here because we don't have a groundtruth data for the last element of the cumsum (see preds for why)
  ##line up epi curve data with cumulative data
  epi.curve <- epi.curve[-1]

  if(length(epi.curve) == 0){
    return(NA)
  }

  ##current fit will be by optim. Setting objective function
  ##takes parameter in order r,p,K...assuming alpha is 1 for now
  ob.func <- function(param) {
    loc.r <- exp(param[1]) #range 0 to inf
    loc.p <- inv_logistic(param[2]) #range 0 to inf
    loc.K <- exp(param[3]) #range 0 to inf

    preds <- loc.r*cum.curve^loc.p * (1 - cum.curve/loc.K)
    rc <- sum(abs(epi.curve - preds))/sum(epi.curve)
    return(rc)
  }

  #call optim wiht som not insane starting values
  tmp <- optim(c(log.r,logistic.p,log.K), ob.func, ...)

  return(list(r = exp(tmp$par[1]), p = inv_logistic(tmp$par[2]), K = exp(tmp$par[3])))
}



##' Function to estimate the progress on an epidemic
##' using the method of Viboud and Chowell, with shared
##' parameters accross epidmeics
##'
##' @param ecs a list of epidemic curves
##' @param cum.curves the cumulative sums to use...allows us to pass
##'        in ecs with missing observations.
##' @param r intrinsic growth parameter
##' @param p "deceleration of growth' parameter
##' @param K final size of the outbreak
##' @param ... additional options to optim
##'
##' @return expected final size of outbreak and params specifying outbreak
final.size.ViboudChowell.pooled <- function(ecs,
                                     cum.curves = sapply(ecs, cumsum),
                                     r =1, p = 0.5,
                                     K = sapply(cum.curves, max),...){

  log.K <- log(K)
  log.r <- log(r)
  logistic.p <- logistic(p)

  for (i in 1:length(ecs)) {
    ## NOTE: when using cum.curve=cumsum(epi.curve), cum.curve is using whatever epi.curve is when it is first used
    cum.curves[[i]] <- cum.curves[[i]][-length(cum.curves[[i]])] ##this was here because we don't have a groundtruth data for the last element of the cumsum (see preds for why)
    ##line up epi curve data with cumulative data
    ecs[[i]] <- ecs[[i]][-1]


    if(length(ecs[[i]]) == 0){
      stop("Empty epidemic curve")
    }
  }

  ##current fit will be by optim. Setting objective function
  ##takes parameter in order r,p,K...assuming alpha is 1 for now
  ob.func <- function(param) {
    loc.r <- exp(param[1]) #range 0 to inf
    loc.p <- inv_logistic(param[2]) #range 0 to inf
    loc.K <- exp(param[3:length(param)]) #range 0 to inf

    rc <- 0

    for (i in 1:length(ecs)) {
      cum.curve <- cum.curves[[i]]
      epi.curve <- ecs[[i]]
      preds <- loc.r*cum.curve^loc.p * (1 - cum.curve/loc.K[i])
      #print(i)
      #print(sum(abs(epi.curve - preds))/sum(epi.curve))
      denom <- sum(epi.curve)
      if(denom==0) {denom<-1}
      rc <- rc + sum(abs(epi.curve - preds))/denom
    }


    return(rc)
  }

    print(log.r)
    print(logistic.p)
    print(range(exp(log.K)))
  #call optim wiht som not insane starting values
  tmp <- optim(c(log.r,logistic.p,log.K), ob.func, ...)

    ##print(tmp)


  return(list(r = exp(tmp$par[1]), p = inv_logistic(tmp$par[2]), K = exp(tmp$par[3:(2+length(ecs))])))
}




##' Function to estimate parameters given K is fixed using K from the gam
##' using the method of Viboud and Chowell.
##'
##' @param epi.curve an epidemic curve
##' @param cum.sum the cumulative sums to use...allows us to pass
##'        in epi.curve with missing observations.
##' @param r intrinsic growth parameter
##' @param p "deceleration of growth' parameter
##' @param K epidemic curves generated from the gam model
##' @param ... additional options to optim
##'
##' @return refitted r and p values

final.size.ViboudChowell.fit.params <- function(epi.curve,
                                                cum.curve = cumsum(epi.curve),
                                                r=1,
                                                p = 0.5,
                                                K = max(cum.curve),...){

  log.K <- log(K)
  log.r <- log(r)
  logistic.p <- logistic(p)

  #print(log.K)
  #print(log.r)
  #print(logistic.p)

  ## NOTE: when using cum.curve=cumsum(epi.curve), cum.curve is using whatever epi.curve is when it is first used
  cum.curve <- cum.curve[-length(cum.curve)] ##this was here because we don't have a groundtruth data for the last element of the cumsum (see preds for why)
  ##line up epi curve data with cumulative data
  epi.curve <- epi.curve[-1]

  if(length(epi.curve) == 0){
    return(100000)
  }

  ##current fit will be by optim. Setting objective function
  ##takes parameter in order r,p,K...assuming alpha is 1 for now
  ob.func <- function(param) {
    loc.r <- exp(param[1])

    loc.p <- inv_logistic(param[2]) #range 0 to 1

    preds <- loc.r*cum.curve^loc.p * (1 - cum.curve/K)
    rc <- sum(abs(epi.curve - preds))/sum(epi.curve)
    return(rc)
  }

  #call optim wiht som not insane starting values
  tmp <- optim(c(log.r,logistic.p), ob.func, ...)
  #cat("fit.parms.r", log.r, ":", logistic.p, "\n")

  return(list(r = exp(tmp$par[1]), p = inv_logistic(tmp$par[2])))
}

##' Function to estimate parameters given K is fixed using K from the gam
##' using the method of Viboud and Chowell using information from all
##' epi curves to create a single set of r and p values.
##'
##' @param epi.curve an list of epidemic curves.
##' @param cum.sum the cumulative sums to use...allows us to pass
##'        in epi.curve with missing observations.
##' @param r intrinsic growth parameter
##' @param p "deceleration of growth' parameter
##' @param K epidemic curves generated from the gam model
##' @param ... additional options to optim
##'
##' @return refitted r and p values
final.size.ViboudChowell.fit.params.pooled <- function(epi.curve,
                                                       cum.curve = lapply(epi.curve,cumsum),
                                                       r=1, p = 0.5,
                                                       K = sapply(cum.curve,max),...){

  log.K <- log(K)
  log.r <- log(r)
  logistic.p <- logistic(p)

  #print(log.K)
  #print(log.r)
  #print(logistic.p)

  ## NOTE: when using cum.curve=cumsum(epi.curve), cum.curve is using whatever epi.curve is when it is first used
  # cum.curve <- cum.curve[-length(cum.curve)] ##this was here because we don't have a groundtruth data for the last element of the cumsum (see preds for why)
  cum.curve <- lapply(cum.curve,function(x){x[-length(x)]}) ##this was here because we don't have a groundtruth
  ##line up epi curve data with cumulative data
  # epi.curve <- epi.curve[-1]
  epi.curve <- lapply(epi.curve,function(x){x[-1]})


  if(length(epi.curve) == 0){
    # return(100000)
    stop("Use at least one epi curve.")
  }
  #Debug only
  lapply(epi.curve,function(x){if(length(x) == 0){stop("Do not use empty epi curves.")}})

  ##current fit will be by optim. Setting objective function
  ##takes parameter in order r,p,K...assuming alpha is 1 for now

  ob.func <- function(param) {
    rc <- 0
    for(i in 1:length(epi.curve)){
      loc.r <- exp(param[1])

      loc.p <- inv_logistic(param[2]) #range 0 to 1

      # preds <- loc.r*cum.curve^loc.p * (1 - cum.curve/K)
      preds <- loc.r*cum.curve[[i]]^loc.p * (1 - cum.curve[[i]]/K[i])

      denom <- sum(epi.curve[[i]])
      if(denom==0) {denom<-1}


      rc <- rc + sum(abs(epi.curve[[i]] - preds))/denom

    }
    return(rc)
  }


    ##call optim wiht som not insane starting values
    print(r,p)
    print(c(log.r,logistic.p))
    print(K)
    tmp <- optim(c(log.r,logistic.p), ob.func, ...)
    print(tmp)
  cat("fit.parms.r", exp(tmp$par[1]), ":", exp(tmp$par[2]), "\n")

    rc <- list(r = exp(tmp$par[1]), p = inv_logistic(tmp$par[2]))
    ##don't let these be 0, sub with  10^-12
    if (rc$r == 0) {rc$r<-10^-12}
    if (rc$p == 0) {rc$p<-10^-12}
  return(rc)

}

##' Function to estimate K using updated parameters
##' using the method of Viboud and Chowell.
##'
##' @param epi.curve an epidemic curve
##' @param cum.sum the cumulative sums to use...allows us to pass
##'        in epi.curve with missing obsrvations.
##' @param r intrinsic growth parameter
##' @param p "deceleration of growth' parameter
##' @param K final size
##' @param ... additional options to optim
##'
##' @return expected final size of outbreak

final.size.ViboudChowell.fit.K <- function(epi.curve,
                                           cum.curve = cumsum(epi.curve),
                                           r=1,
                                           p=0.5,
                                           K = max(cum.curve), ...){

  logistic.p <- logistic(p)
  log.r <- log(r)
  log.K <- log(K)

  #cat("passed.parms.r", log.r, ":", logistic.p, "\n")

  ## NOTE: when using cum.curve=cumsum(epi.curve), cum.curve is using whatever epi.curve is when it is first used
  cum.curve <- cum.curve[-length(cum.curve)] ##this was here because we don't have a groundtruth data for the last element of the cumsum (see preds for why)
  ##line up epi curve data with cumulative data
  epi.curve <- epi.curve[-1]

  if(length(epi.curve) == 0){
    #return(100000)
    stop("Use at least one epi curve.")
  }

  ##current fit will be by optim. Setting objective function
  ##takes parameter in order r,p,K...assuming alpha is 1 for now

  ob.func <- function(param) {
    loc.K <- exp(param[1]) #range 0 to inf

    preds <- r*cum.curve^p * (1 - cum.curve/loc.K)



    if(length(epi.curve) < 5){
      warning("This needs longer epi curves")
      return(0)
    }

    #rc <- sum(((log(epi.curve+1) - log(preds+1))^2))  #+ loc.K^.1875
    rc <- sum(abs(epi.curve - preds))/sum(epi.curve) +
        (abs(max(cum.curve)-loc.K)^.01-1) #weak penalty for additional cases
    ##cat("**", loc.K,":", K,":", rc,"\n")#Debug

    return(rc)
  }



  #call optim wiht som not insane starting values
  tmp <- optim(c(log.K), ob.func, method = "Brent",
               lower = log(max(cum.curve, na.rm = T)) ,
               upper = log(7*10^9), ...)
  #tmp <- optimize(ob.func, method = "L-BFGS-B", lower = 0 , upper = log(7*10^9), ...)

  #print(tmp) ##DEBUG

  #if (exp(tmp$par[1])>10^7) stop("Exploding K")
  return(list(K = exp(tmp$par[1])))
}

##' Function gets the error compared to the final estimate
##' for final size estimates over the entire epidemic
##' curve
##'
##' @param epi.curve the epidemic curve to be analyzed
##' @param ... additional parameters to the fitting
##'
##' @return a "time series" of errors

ts.err.final.size.ViboudChowell <- function(epi.curve, ...) {


}

##'Function runs a jacknife calculation on the final size
##'for the Viboud Chowell model
##'
##' @param epi.curve
##' @param ... additional parameters to optim
##'
##' @return

jackknife.final.size.ViboudChowell <- function(epi.curve,...) {

  rc <- vector(length=length(epi.curve)-1)
  cum.curve<- cumsum(epi.curve)

  for (i in 2:length(epi.curve)) {
    tmp.cum <- cum.curve[-(i-1)]
    tmp.epi <- epi.curve[-i]
    rc[i-1] <- final.size.ViboudChowell(tmp.epi,tmp.cum, ...)
  }

  return(rc)
}

##' Function generates a bunch of epidemic curves of different
##' percent done based on a ViboudChowell model given values of
##' r and p and a series of final sizes.
##'
##' @param pct a vector of percent of epidemic done of the same length
##'     as the number we should generate.
##' @param K a vector of final epidemic sizes
##' @param r the growth rate
##' @param p the extent the growthrate is super or sub exponential
##'
##' @return a list of epidemic curves
gen.epicurves.ViboudChowell <- function (pct, K, r, p) {
  rc <- list()

  for (i in 1:length(pct)) {
    ## for each epidemic we want to generate, start with
    ## a single case and then loop until we generate
    ## the appropriate percentage of the final size.
    ec <- (1)
    while (sum(ec)<K[i]*pct[i]) {
        ec <- c(ec,rpois(1, r*sum(ec)^p*(1-sum(ec)/K[i])))
        ##ec <- c(ec,r*sum(ec)^p*(1-sum(ec)/K[i]))#debug
    }
    rc[[i]] <- ec
  }

  return(rc)
}


##' function generates a bunch of epidemic curves of a different percent done
##' based on a stochastic discrete time SIR model. Presume epidemic starts with a single
##' case.
##'
##' @param pct a vector of the percent epidemic done of th esmae length we should generate
##' @param beta a vector of betas for each simulation
##' @param N a verctor of population sizes
##' @param gamma a vector of the recovery rates, or a sigle shared value
##'
##' @return a list of epidemic curves and a vector of final sizes.
gen.epidemic.curves.SIR <- function(pct, beta, N, gamma) {

    if (length(beta)==1) {beta <- rep(beta, length(pct))}
    if (length(gamma)==1) {gamma <- rep(gamma, length(pct))}
    if (length(N)==1) {gamma <- rep(N, length(pct))}

    rc_ecs <- list()
    rc_fs <- c()


    for (i in 1:length(pct)) {
        #print(i)
        ## generate the epidemic curve.
        repeat {
            S <- N[i]-1
            I <- 1
            R <- 0

            curve <-1

            while (I > 0) {
                infs <- rbinom(1,S, 1-exp(-I*beta[i]/N[i]))
                recovs <- rbinom(1,I, 1-exp(-gamma[i]))
                S <- S - infs
                I <- I + infs - recovs
                R <- R + recovs

                curve <- c(curve, infs)
            }

            if (sum(curve) > 10) { #eliinate stochastic fadeouts
                break
            }
        }

        ## truncate it by pct
        fs <- sum(curve)

        target_sz <-fs * pct[i]
        ind <- max(c(which(cumsum(curve)<=target_sz),2)) #ust have at least 2 obse


        rc_ecs[[i]] <- curve[1:ind]
        rc_fs <- c(rc_fs, fs)

    }

    return(list(ecs=rc_ecs, fs=rc_fs))
}

##' Perform an MCMC to estimate percent of epidemic complete from
##' a combination of covariates and predictor variables. Uses GAMs
##' to fit covariate relationship, and assumes ViboudChowell
##' underlying model.
##'
##' @param epi.curves all the epi curves we are going
##' @param x data frame of covariates
##' @param f formula to add to use in the GAM. Assume outcome variable
##'        is called K
##' @param mcmc.iter the number of interations to run
##'
##' @return the samples from the MCMC

simple.EPInf.mcmc <- function (epi.curves, x, f, mcmc.iter = 1000) {
  require(mgcv)

  ##first determine priors for the percent done.
  expected.K <- rep(0, length(epi.curves))
  for (i in 1:length(epi.curves)) {
    expected.K[i] <-
      final.size.ViboudChowell(epi.curves[[i]])
  }

  pcts <- sapply(epi.curves, sum)/expected.K
  ##convert pcts above 1 to fractionally less than 1.
  pcts[pcts>1] <- 0.999999999

  cur.pcts <- log(pcts/(1-pcts))

  totals <- sapply(epi.curves, sum)


  ##set up the matrix for MCMC return
  rc <- matrix(nrow=mcmc.iter, ncol=length(pcts))

  ##fit model at starting parameters. Note that here we can just
  ##use K
  x$K <- sqrt(expected.K)
  mdl <- gam(formula(f),data=x)

  ##calculate the likelihood of the observed
  ##given pcts and expected K.
  cur.ll <- sum(dpois(totals,logistic(cur.pcts)*
                        predict(mdl)^2, log=T)) +
    sum(dpois(round(expected.K), expected.K, log=TRUE)) #clearly part of the problem

  rc[1,] <- cur.pcts

  ##Do the MCMC loops
  for (i in 2:mcmc.iter) {
    ##propose new pcts for a random place
    prop.pcts <- cur.pcts
    ind <- sample(length(prop.pcts),1)
    prop.pcts[ind] <- cur.pcts[ind] + rnorm(1, 0,1)
    prop.K <- totals/logistic(prop.pcts)

    ##refit model
    x$K <- sqrt(prop.K)
    mdl <- gam(formula(f),data=x)


    ##calculate proposal likelihood
    prop.ll <- sum(dpois(totals,logistic(prop.pcts)*
                           predict(mdl)^2, log=T)) +
      sum(dpois(round(prop.K), expected.K, log=TRUE))

    if ((prop.ll - cur.ll) > log(runif(1))) {
      ##accept
      cur.ll <- prop.ll
      cur.pcts <- prop.pcts
    }

    rc[i,] <- cur.pcts

  }

  return(rc)

}

##' Simple compare function that will be used in simple.epiInf.EM to generate
##' the absolute error of the observed and predicted percent complete of the
##' epidemic curve
##' @param this_iter the current iteration of the loop
##' @param last_iter the previous iteration of the loop
##'
##' @return the absolute error of percent complete between the iterations

compare <- function(this_iter, last_iter){
  # if(!(('pcts' %in% names(this_iter)) & ('pcts' %in% names(last_iter)))){
  #   stop("pcts not found")
  # }
  #
  # if(length(this_iter$pcts) != length(last_iter$pcts)){
  #   stop("arguments have different lengths")
  # }

    K.diff <- abs(this_iter$K - last_iter$K)
    K.stat.diff <- abs(this_iter$K.stat - last_iter$K.stat)

  return(max(sum(K.diff), sum(K.stat.diff)))
}

##' Initialization function. Estimates final size and
##' parameters based on epidemic curve that will be used to start
##' EM approach.
##' @param epi.curve the epidemic curve to be analyzed
##'
##' @return list of final sizes and percent done of epidemic

initialize.em <- function(epi.curves){

  res <- final.size.ViboudChowell.pooled(epi.curves, method="BFGS")

  return(list(K=res$K, r=res$r,p=res$p))
}


##' Simple EM approach to estimating epidemic progress.
##' Basic algorithm is to initialize using epi-curve data
##' and then interate between fitting model and fitting
##' epi curve data using pooled estimates until convergance.
##' Using a GAM model to start.
##'
##'  @param epi.curves the individual epi curves
##'  @param x data frame of covariates
##'  @param f formula to use in GAM
##'  @param max.iter the maxiumum number of iterations performed
##'  @param threshold the desired precision of output
##'
##'  @return a vector of estimated final sizes
##'
simple.epiInf.EM <- function(epi.curves, x, f, max.iter = 100, threshold = 10^-5){
  require(mgcv)
  #browser()
  ## initialize using information from epidemic curves
  this_iter <- initialize.em(epi.curves)
  #print(this_iter)

  ## We want to use the total observed cases as a lower bound for final size
  #lower_K <- sapply(epi.curves,sum)

  ##Loop until ending critera is met
  iter <- 0
  #print(iter)
  ## make sure values are above the threshold
  last_iter <- list(K = this_iter$K+2*threshold,
                    #pcts=this_iter$pcts+2*threshold,
                    r = this_iter$r+2*threshold,
                    p = this_iter$p+2*threshold)

  Khist <- NULL
  Khist.stat <- NULL
  par.hist <- NULL

  while(
    (compare(last_iter, this_iter) >= threshold) &
    (iter < max.iter)){
    # browser()
    ## Re-starting loop
    last_iter <- this_iter


    ##Get the length of the epi curves.
    epi.curve.length <- c()
    for(i in 1:length(epi.curves)){
      curve.length <- length(epi.curves[[i]])
      epi.curve.length[i] <- curve.length
    }


    ## fit statistical model (use final size for covariates)
    ## where currently model is 'K~x'
    x$K <- round(this_iter$K) #Rounding becuase we are uding poisson
    # x$K = this_iter$K
    #mdl <- gam(formula(f), data = x)
    #mdl <- gam(formula(f), data = x, weights = epi.curve.length)
    mdl <- gam(formula(f), data = x, family = poisson(), weights = epi.curve.length)

    ## predict model
    K.stat <- predict(mdl,type='response')
    #cat("statistical prediction", K.stat, "\n")
    #print(cbind(x$K,K.stat))


    ## re-estimate parameters
    ## only want final percentage complete
    #expected.r <- rep(0, length(epi.curves))


        tmp <- final.size.ViboudChowell.fit.params.pooled(epi.curves,
                                                          K = K.stat,
                                                          r = this_iter$r,
                                                          p = this_iter$p,
                                                          method="BFGS")

    #cat(this_iter$r,"->",tmp$r," : ", this_iter$p, "->", tmp$p, "\n")
    expected.p <- tmp$p
    expected.r <- tmp$r


    ## estimate final size based on epidemic curve
    expected.K <- c()

    for(i in 1:length(epi.curves)) {
        tmp <- final.size.ViboudChowell.fit.K(epi.curves[[i]], r = expected.r, p = expected.p, K = K.stat[[i]])

      expected.K[i] <- tmp$K
    }

    this_iter$K <- expected.K
    this_iter$r <- expected.r
    this_iter$p <- expected.p
    this_iter$K.stat <- K.stat

    Khist <-rbind(Khist, expected.K)
    Khist.stat <- rbind(Khist.stat, K.stat)
    par.hist <- rbind(par.hist, c(this_iter$r, this_iter$p))

    # if(iter %% 10 == 0){
    #   cat("model prediction ",iter,":", expected.K, "\n")
    #   print(summary(mdl))
    #   print(paste("iter =" ,iter, sep=""))
    #   print("K")
    #   print(this_iter$K)
    #   print("r")
    #   print(this_iter$r)
    #   print("p")
    #   print(this_iter$p)
    #   print(K.stat)
    # }

    #print(compare(last_iter, this_iter))
    #print(last_iter)
    #print(this_iter)
    #this_iter$pcts <- mapply(x=epi.curves,y=this_iter$K,function(x,y){x[length(x)]/y})

    #this_iter$pcts[this_iter$pcts > 1] <- 0.999999999
        iter = iter + 1

        cat("Err : ",compare(last_iter, this_iter),"\n")
  }
  print(iter)
  #return(this_iter$pcts)
  rc <- this_iter
  rc$Khist <- Khist
  rc$Khist.stat <- Khist.stat
  rc$par.hist <- par.hist
  return(rc)


}


##' Algorithm does the EM fit of viboud chowell and a gam, but without pooling parameters.
##'
##'

##' Initialization function. Estimates final size
##' based on cumulative epidemic curve that will be
##' used to start the statistical model approach
##'
##' @param epi.curve the epidemic curve to be analyzed
##'
##' @return list of final sizes of the epidemic

initialize_stat <- function(epi.curves){
  K <- sapply(epi.curves, sum)
  return(list(K=K))
}

##' Naive statistical approach to estimating epidemic progress.
##' Basic algorithm is to initialize using epi-curve data
##' and then generate final size from statistical model
##'
##'  @param epi.curves the individual epi curves
##'  @param x data frame of covariates

stat_inference <- function(epi.curves, x){

  fit <- initialize_stat(epi.curves)

  x$K <- round(fit$K)

  mdl <- glm(log(K) ~ x , data = x)

  stat_pred <- predict(mdl,type='response')

  fit$stat_pred <- exp(stat_pred)

  rc <- fit
  return(rc)

}




################### MORE GENERAL FRAMEWORK#################

##' Functoin that fits a general epidemic and statistical model using an EM
##' algorithm.
##'
##' @param epi.curves the individual epidemic curves to fit
##' @param x data frame of covariates. First columns assumed to contain K initialized for the odel fits
##' @param stat.mdl.fit a function that fits the statistical model with the following parameters and returns a fir model:
##'               - x: a data frame of covariates that must include the final size column K a
##' @param epi.mdl.fit a function that calls the epi model with the following parameter and returns a fit model:
##'               - epi.curves: the epidemic curves.
##'               - K: the projected fornal sizes
##'               - prev.mdl : the previous model. should behave when this is null
##' @param stat.mdl.pred: a function that takes in a statisitcal model and returns a vector of predicted final sizes
##' @param epi.mdl.pred: a fuctnion that takes in the epdiemic model and and returns a vector of predicted final sizes
##' @param max.iter the maximum number of iterations to run
##' @param threshold the desired precision of the putput
##'
##'
##' @return a vector of estimated final sizes
##'
epiInf.EM <- function (epi.curves, x,
                       stat.mdl.fit, epi.mdl.fit,
                       stat.mdl.pred, epi.mdl.pred,
                       max.iter=100, threshold = 10^-5) {


    iter <- 0 #keep track of iterations

    ##for the first loop through, make sure that we are abover threshold
    iter.diff <- 2*threshold *2

    ##matrices for holding results.
    K <- matrix(nrow=max.iter, ncol= length(epi.curves))
    K.stat <- matrix(nrow=max.iter, ncol= length(epi.curves))

    ##previous epi model
    prev.epi.mdl <- NULL

    ## Loop until endinc criteria is met
    while ((iter.diff >= threshold) &
           (iter<max.iter))  {

               iter <- iter+1


               ## Fit the statistical model on htis iteration
               fit.stat.mdl <-stat.mdl.fit(x)
               K.stat[iter,] <- stat.mdl.pred(fit.stat.mdl)


               #print(K.stat[iter,])

               ## Fit the epidemic model using the stat stuff
               fit.epi.mdl <- epi.mdl.fit(epi.curves, K.stat[iter,], prev.mdl=prev.epi.mdl)
               prev.epi.mdl <- fit.epi.mdl

               ##print(cbind(fit.epi.mdl$peak.time, fit.epi.mdl$spread, fit.epi.mdl$K)) ##DEBUG

               K[iter,] <- epi.mdl.pred(fit.epi.mdl)

               ## Update the dta matrix with these Ks
               x$K <- K[iter,]

               ## update the iter differential if this is the second iteratoin or beyond
               if (iter>1) {
                   ##iter.diff <- max(sum(abs(K[iter-1,]-K[iter,])),
                   ##                 sum(abs(K.stat[iter-1,]-K.stat[iter,])))
                   iter.diff <- sum(abs(K[iter-1,]-K[iter,]))/length(K[iter,])
                   #print(cbind(K[iter-1,],K[iter,]))

               }

               ##print(cbind(x$K, K[iter,]))
               ##hist(K.stat[iter,]- K[iter,])
               ##print(which(abs(K.stat[iter,]- K[iter,])>10000))
               #cat(iter, ":",iter.diff ,":", range(K[iter,]),":",
               #    range(K.stat[iter,]),"\n")
           }


    ##return a list with the two Ks and the final Ks
    return(list(K=K[iter,],
                K.stat=K.stat[iter,],
                Khist = K[1:iter,],
                Khist.stat = K.stat[1:iter,]))
}


##' Function for doing random forrest fit of the K data appropriate for
##' passing in to epiInf.EM
##'
##' @param x the data frame to fit basedd on
##'
##' @return a fit random forrest model
##'
stat.mdl.rf.fit <- function(x) {
    require(randomForest)

    y <- x$K
    x <- as.matrix(subset(x, select=-K))

    rc <- randomForest(x=x, y=y)

    return(rc)
}



##'Function for doing the predict for duing the predict for the
##' results from a stat.mdl.rf.fit
##'
##' @param mdl the fit model
##'
##' @return a vector of predicted final sizes
##'
stat.mdl.rf.pred <- function(mdl) {
    return(predict(mdl))
}



##' Function for doing random forrest fit of the K data appropriate for
##' passing in to epiInf.EM
##'
##' @param x the data frame to fit basedd on
##'
##' @return a fit random forrest model
##'
stat.mdl.rpart.fit <- function(x) {
    require(rpart)

    f <- formula(paste0("K~",paste(names(subset(x,select=-K)),collapse="+")))
    rc <- rpart(f, data=x)

    return(rc)
}



##'Function for doing the predict for duing the predict for the
##' results from a stat.mdl.rf.fit
##'
##' @param mdl the fit model
##'
##' @return a vector of predicted final sizes
##'
stat.mdl.rpart.pred <- function(mdl) {
    return(predict(mdl))
}



##' Function for doing the fit for a viboud chowell model of the epidemic curves.
##' Parameters are considered to be drawn from a shared distribution. Fit is done via optim
##'
##' @param ecs the epidemic curves
##' @param K the Ks
##'
##' @return a vector of parameters for the fit model
##'
epi.mdl.ViboudChowell.fit <- function (ecs, K) {

    ##make some starting values for p and r with a touch of noise
    logistic.p.intercept <- 0
    logistic.p <- rnorm(length(ecs),0,1)

    log.r.intercept <- 0
    log.r <- rnorm(length(ecs),0,1)



    ## Calculate the cumulative incidence curves
    cum.curve = lapply(ecs,cumsum)
    cum.curve <- lapply(cum.curve,function(x){x[-length(x)]}) ##drop the last cumsum observation as it does not contibute
    epi.curve <- lapply(ecs,function(x){x[-1]}) ##first observed epi curve also does not contribute

    lapply(epi.curve,function(x){if(length(x) == 0){stop("Do not use empty epi curves.")}})


    ## will fit by optim using poisson error function at each step with shared priors.
    ## param vector is structured as follows....first parameter is shared r
    ## parameters 2...(length(ec)+1) are individual rs
    ## parameters (length(ec)+2 is shard p
    ## parameters length(ec)+3...2*length(ec)+2 are inividual ps
    ob.func <- function(param) {
        rc <- 0
        for(i in 1:length(epi.curve)){
            loc.r <- exp(param[1+i])
            loc.p <- inv_logistic(param[length(ecs)+2+i]) #range 0 to 1

            ## preds <- loc.r*cum.curve^loc.p * (1 - cum.curve/K)

            preds <- loc.r*cum.curve[[i]]^loc.p * (1 - cum.curve[[i]]/K[i])
            preds[preds<.1] <- .1
            #print(length(param))
            #cat("locp :", loc.p, "locr :", loc.r,"\n")
            #print(cbind(preds,epi.curve[[i]], dpois(epi.curve[[i]],preds)))
            log.prob <- sum(dpois(epi.curve[[i]], preds, log=TRUE)) +
                dnorm(param[1+i],param[1], 3) + ##magic number of variance
                dnorm(param[length(ecs)+2+i], param[length(ecs)+2], 3) ##magic number
            #cat(i,":",log.prob,"\n")

            rc <- rc - log.prob


        }
        return(rc)
    }

    tmp <- optim(c(log.r.intercept, log.r, logistic.p.intercept, logistic.p), ob.func,
                 method="CG")
    return(list(r=exp(tmp$par[(1+(1:length(ecs)))]), p=inv_logistic(tmp$par[(2+length(ecs)+(1:length(ecs)))]), ecs=ecs,
                fit.Ks=K))

}



##' Funciton for doing the "predict" step for the viboud cholel model
##' Essentially takes the p and r parameters and fits K given those parameters
##'
##' Has some weak schrinkage of K towards a single value
##'
##' @param pars the parameters from a model fit given K
##'
##' @return a vector of Ks
##'
epi.mdl.ViboudChowell.pred <- function(pars) {
    log.K <- log(pars$fit.Ks)
    ecs <- pars$ecs

    cum.curve = lapply(ecs,cumsum)
    cum.curve <- lapply(cum.curve,function(x){x[-length(x)]}) ##drop the last cumsum observation as it does not contibute
    epi.curve <- lapply(ecs,function(x){x[-1]}) ##first observed epi curve also does not contribute

    lapply(epi.curve,function(x){if(length(x) == 0){stop("Do not use empty epi curves.")}})


    ## Joint fitting of Ks
    ob.func <- function(param) {
        rc <- 0
        for(i in 1:length(epi.curve)){
            loc.r <- pars$r[i]
            loc.p <- pars$p[i]
            loc.a <- pars$a[i]
            K <- exp(param[i])

            ## preds <- loc.r*cum.curve^loc.p * (1 - cum.curve/K)
            preds <- loc.r*cum.curve[[i]]^loc.p * (1 - (cum.curve[[i]]/K)^loc.a)
            preds[preds<.1] <- .1
            #print(length(param))
            #cat("locp :", loc.p, "locr :", loc.r," loc.K:", K, "\n")
            #print(preds)
            log.prob <- sum(dpois(epi.curve[[i]], preds, log=TRUE))#+
                #dnorm(log(K), mean(param), 2, log=TRUE)

            rc <- rc - log.prob
        }

        return(rc)
    }

    ##tmp <- optim(log.K, ob.func,control=list(maxit=5000))
    tmp <- optim(log.K, ob.func, method="Nelder-Mead", control=list(maxit=10000))

    print(tmp)

    return(exp(tmp$par))

}




##' Function for doing the first fit for a viboud chowell model of the epidemic curves.
##' Parameters are considered to be drawn from a shared distribution.
##'
##' Fit is done via optim
##'
##' @param ecs the epidemic curves
##'
##' @return a vector of Ks
##'
epi.mdl.ViboudChowell.init <- function (ecs) {

    ##make some starting values for p and r with a touch of noise
    logistic.p.intercept <- 0
    logistic.p <- rnorm(length(ecs),0,1)

    log.r.intercept <- 0
    log.r <- rnorm(length(ecs),0,1)

    log.K <- log(2*sapply(ecs,sum))

    log.a.intercept <- 0
    log.a <- rnorm(length(ecs),0,1)


    ## Calculate the cumulative incidence curves
    cum.curve = lapply(ecs,cumsum)
    cum.curve <- lapply(cum.curve,function(x){x[-length(x)]}) ##drop the last cumsum observation as it does not contibute
    epi.curve <- lapply(ecs,function(x){x[-1]}) ##first observed epi curve also does not contribute


    lapply(epi.curve,function(x){if(length(x) == 0){stop("Do not use empty epi curves.")}})


    ## will fit by optim using poisson error function at each step with shared priors.
    ## param vector is structured as follows....first parameter is shared r
    ## parameters 2...(length(ec)+1) are individual rs
    ## parameters (length(ec)+2 is shard p
    ## parameters length(ec)+3...2*length(ec)+2 are inividual ps
    ## parameter 2*length(ec)+3...3*length(ec) are the Ks
    ob.func <- function(param) {
        rc <- 0
        for(i in 1:length(epi.curve)){
            loc.r <- exp(param[1+i])
            loc.p <- inv_logistic(param[length(ecs)+2+i]) #range 0 to 1
            loc.K <- exp(param[2*length(ecs)+2+i])
            loc.a <- exp(param[3*length(ecs)+3+i])

            ## preds <- loc.r*cum.curve^loc.p * (1 - cum.curve/K)

            preds <- loc.r*cum.curve[[i]]^loc.p * (1 - (cum.curve[[i]]/loc.K)^loc.a)


            preds[preds<.1] <- .1


            log.prob <- sum(dpois(epi.curve[[i]], preds, log=TRUE)) +
                dnorm(param[1+i],param[1], 3, log=T) + ##magic number of variance
                dnorm(param[length(ecs)+2+i], param[length(ecs)+2], 3, log=T) +
                dnorm(param[3*length(ecs)+3+i], param[3*length(ecs)+3], 3, log=T)# +
                ##dnorm(log10(loc.K),0,5, log=T)



            ## if(is.nan(log.prob)) {
            ##      print(cbind(preds,epi.curve[[i]], dpois(epi.curve[[i]],preds, log=T)))
            ##      cat("locp :", loc.p, "locr :", loc.r,"logprob:", log.prob,"\n")
            ## }

            rc <- rc - log.prob

        }
        #print(rc)

        return(rc)
    }

    tmp <- optim(c(log.r.intercept, log.r, logistic.p.intercept, logistic.p, log.K,
                   log.a.intercept, log.a), ob.func,
                  control=list(maxit=10000))
    print(tmp)
    return(list(r=exp(tmp$par[(1+(1:length(ecs)))]),
                p=inv_logistic(tmp$par[(2+length(ecs)+(1:length(ecs)))]),
                K=exp(tmp$par[2*length(ecs)+2+1:length(ecs)]),
                a=exp(tmp$par[3*length(ecs)+3+1:length(ecs)])))

}



##' Function for doing the first fit for a viboud chowell model of the epidemic curves.
##' Parameters are fit independently for each epidemic
##'
##' Fit is done via optim
##'
##' @param ecs the epidemic curves
##'
##' @return a vector of Ks
##'
epi.mdl.ViboudChowell.init2 <- function (ecs) {

    ##make some starting values for p and r with a touch of noise
    logistic.p <- rnorm(length(ecs),0,1)

    log.r <- rnorm(length(ecs),0,1)

    log.K <- log(2*sapply(ecs,sum))

    log.a <- rnorm(length(ecs),0,1)

    ## Calculate the cumulative incidence curves
    cum.curve = lapply(ecs,cumsum)
    cum.curve <- lapply(cum.curve,function(x){x[-length(x)]}) ##drop the last cumsum observation as it does not contibute
    epi.curve <- lapply(ecs,function(x){x[-1]}) ##first observed epi curve also does not contribute


    lapply(epi.curve,function(x){if(length(x) == 0){stop("Do not use empty epi curves.")}})



    ob.func <- function(param) {
        rc <- 0
        loc.r <- exp(param[1])
        loc.p <- inv_logistic(param[2]) #range 0 to 1
        loc.K <- exp(param[3])
        loc.a <- exp(param[4])

        preds <- loc.r*cum.curve[[i]]^loc.p * (1 - (cum.curve[[i]]/loc.K)^loc.a)


        preds[preds<.1] <- .1


        log.prob <- sum(dpois(epi.curve[[i]], preds, log=TRUE)) +
            dnorm(log10(loc.K),0,5, log=T)

        rc <- - log.prob

        return(rc)
    }

    rc <- list()

    for (i in 1:length(ecs)) {
        tmp <- optim(c(log.r[i], logistic.p[i],
                       log.K[i], log.a[i]), ob.func,
                     control=list(maxit=10000))
        rc$converged <- c(rc$converged,tmp$convergence)
        rc$r <- c(rc$r, exp(tmp$par[1]))
        rc$p <- c(rc$p, inv_logistic(tmp$par[2]))
        rc$K <- c(rc$K, exp(tmp$par[3]))
        rc$a <- c(rc$a, exp(tmp$par[4]))

        }

    return(rc)


}


##' Funciton for doing the "predict" step for the viboud cholel model
##' Essentially takes the p and r parameters and fits K given those parameters
##'
##' Has some weak schrinkage of K towards 0e
##'
##' @param pars the parameters from a model fit given K
##'
##' @return a vector of Ks
##'
epi.mdl.ViboudChowell.pred2 <- function(pars) {
    log.K <- log(pars$fit.Ks)
    ecs <- pars$ecs

    cum.curve = lapply(ecs,cumsum)
    cum.curve <- lapply(cum.curve,function(x){x[-length(x)]}) ##drop the last cumsum observation as it does not contibute
    epi.curve <- lapply(ecs,function(x){x[-1]}) ##first observed epi curve also does not contribute

    lapply(epi.curve,function(x){if(length(x) == 0){stop("Do not use empty epi curves.")}})




    ##Individual fitting of Ks
    ob.func <- function(param, loc.r, loc.p, loc.a, cum.curve, epi.curve) {
        K <- param
        preds <- loc.r*cum.curve^loc.p * (1 - (cum.curve/K)^loc.a)
        preds[preds<.1] <- .1

        log.prob <- sum(dpois(epi.curve, preds, log=TRUE)) +
             dnorm(log10(K),0,9, log=T)

        return(-log.prob)
    }

    rc <- rep(NA, length(log.K))


    for (i in 1:length(rc)) {
        tmp <- optimize(ob.func, c(sum(ecs[[i]]), 10^7),
                        loc.r=pars$r[i], loc.p=pars$p[i],
                        loc.a=pars$a[i],
                        cum.curve = cum.curve[[i]],
                        epi.curve=epi.curve[[i]])
        rc[i] <- tmp$minimum
    }

    return(rc)
}




##' Function for doing the fit for a viboud chowell model of the epidemic curves.
##' Parameters are considered to be independent for each epidem. Fit is done via optim
##'
##' @param ecs the epidemic curves
##' @param K the Ks
##' @param prev.mde an optional argument with the previous model
##'
##' @return a vector of parameters for the fit model
##'
epi.mdl.ViboudChowell.fit2 <- function (ecs, K, prev.mdl=NULL) {

    ##make some starting values for p and r with a touch of noise
    if (is.null(prev.mdl)) {
        logistic.p <- rnorm(length(ecs),0,1)
        log.r <- rnorm(length(ecs),0,1)
        log.a <- rnorm(length(ecs),0,1)
    } else {
        logistic.p <- prev.mdl$logistic.p
        log.r <- prev.mdl$log.r
        log.a <- prev.mdl$log.a
    }



    ## Calculate the cumulative incidence curves
    cum.curve = lapply(ecs,cumsum)
    cum.curve <- lapply(cum.curve,function(x){x[-length(x)]}) ##drop the last cumsum observation as it does not contibute
    epi.curve <- lapply(ecs,function(x){x[-1]}) ##first observed epi curve also does not contribute


    lapply(epi.curve,function(x){if(length(x) == 0){stop("Do not use empty epi curves.")}})



    ob.func <- function(param, loc.K) {
        rc <- 0
        loc.r <- exp(param[1])
        loc.p <- inv_logistic(param[2]) #range 0 to 1
        loc.a <- exp(param[3])

        preds <- loc.r*cum.curve[[i]]^loc.p * (1 - (cum.curve[[i]]/loc.K)^loc.a)


        preds[preds<.1] <- .1


        log.prob <- sum(dpois(epi.curve[[i]], preds, log=TRUE)) +
            dnorm(log10(loc.K),0,9, log=T)

        rc <- - log.prob

        return(rc)
    }

    rc <- list()

    for (i in 1:length(ecs)) {
        tmp <- optim(c(log.r[i], logistic.p[i],
                       log.a[i]), ob.func,
                     loc.K = K[i],
                     control=list(maxit=10000))
        rc$converged <- c(rc$converged,tmp$convergence)
        rc$log.r <- c(rc$log.r, tmp$par[1])
        rc$logistic.p <- c(rc$logistic.p, tmp$par[2])
        rc$log.a <- c(rc$log.a, tmp$par[3])

    }

    rc$r <- exp(rc$log.r)
    rc$p <- inv_logistic(rc$logistic.p)
    rc$a <- exp(rc$log.a)

    rc$fit.Ks <- K
    rc$ecs <- ecs

    return(rc)


}



##' Version of the epidemiologic model that fits a normal distribution to an epidemic curve.
##' INitialization fits all parameters, including final size, peak time (i.e., mean) and
##' spread (i.e., standard deviation)
##'
##' @param ecs the epidemic curves
##'
##' @return a list with final size, peak time and spread
##'
epi.mdl.normcurve.init <- function(ecs) {

    fs <- 2*sapply(ecs, sum) #start at twice the observed cases
    peak.time <- sapply(ecs, length) #the consistent choice for peak time is the length of the EC
    spread <- peak.time/3 #the consistent choice for peak time and size
    converged <- rep(0, length(ecs))

    ##objective function. POisson error on pedicted incdience.
    ob.func <- function(par, ec) {
        loc.fs <- exp(par[1])
        loc.sd <- exp(par[3])
        pred <- loc.fs * (pnorm(1:length(ec), par[2], loc.sd) -  pnorm((1:length(ec))-1, par[2], loc.sd))


        log.prob <- sum(dpois(ec, pred, log=TRUE))  +
            dnorm(log10(loc.fs), 0,1, log=TRUE)

        if(is.nan(log.prob)) {
            print(par)
            print(ec)
        }

        return(-log.prob)
    }


    for (i in 1:length(ecs)) {
        ec <- ecs[[i]]


        #print(c(fs[i], peak.time[i], spread[i]))
        tmp <- optim(c(log(fs[i]), peak.time[i], log(spread[i])), ob.func, ec=ecs[[i]])

        converged[i] <- tmp$convergence
        fs[i] <- exp(tmp$par[1])
        peak.time[i] <- tmp$par[2]
        spread[i] <- exp(tmp$par[3])
    }

    return(list(K=fs, peak.time = peak.time, spread=spread, converged=converged))

}



##' Version of the epidemiologic model that fits a normal distribution to an epidemic curve.
##' fit fits the
##'
##' @param ecs the epidemic curves
##' @param K the values of K
##' @param prev.mdl the previous model, can be null
##'
##' @return a list with final size, peak time and spread
##'
epi.mdl.normcurve.fit <- function(ecs, K, prev.mdl=NULL) {

    fs <- K #start at twice the observed cases
    peak.time <- sapply(ecs, length) #the consistent choice for peak time is the length of the EC

    if (is.null(prev.mdl)) {
        peak.time <- sapply(ecs, length) #the consistent choice for peak time is the length of the EC
        spread <- peak.time/3 #the consistent choice for peak time and size
    } else {
        peak.time <- prev.mdl$peak.time
        spread <- prev.mdl$spread
    }

    converged <- rep(0, length(ecs))

    ##objective function. POisson error on pedicted incdience.
    ob.func <- function(par, ec, fs) {
        loc.sd <- exp(par[2])
        pred <- fs * (pnorm(1:length(ec), par[1], loc.sd) -  pnorm((1:length(ec))-1, par[1], loc.sd))

        log.prob <- sum(dpois(ec, pred, log=TRUE))

        if (is.nan(log.prob)) {
            print(par)
            print(ec)
        }

        return(-log.prob)
    }


    for (i in 1:length(ecs)) {
        ec <- ecs[[i]]


        #print(c(fs[i], peak.time[i], spread[i]))
        tmp <- optim(c(peak.time[i], log(spread[i])), ob.func, ec=ecs[[i]], fs=fs[i])

        converged[i] <- tmp$convergence
        peak.time[i] <- tmp$par[1]
        spread[i] <- exp(tmp$par[2])
    }

    return(list(K=fs, peak.time = peak.time, spread=spread, converged=converged, ecs=ecs))

}



##' Version of the epidemiologic model that fits a normal distribution to an epidemic curve.
##' INitialization fits all parameters, including final size, peak time (i.e., mean) and
##' spread (i.e., standard deviation)
##'
##' @param ecs the parameters of the fit model
##'
##' @return a list with final size, peak time and spread
##'
epi.mdl.normcurve.pred <- function(pars) {

    #everything starts at the previous estimates for everything
    fs <- pars$K
    peak.time <- pars$peak.time
    spread <- pars$spread
    converged <- rep(0, length(pars$ecs))


    ##objective function. POisson error on pedicted incdience.
    ob.func <- function(par, ec, peak.time, spread) {

        pred <- par * (pnorm(1:length(ec), peak.time, spread) -  pnorm((1:length(ec))-1, peak.time, spread))

        log.prob <- sum(dpois(ec, pred, log=TRUE)) +
            dnorm(log10(par),0, 1, log=TRUE)

        if(is.nan(log.prob)) {
            print(par)
            print(peak.time)
            print(spread)
            print(ec)
        }

        return(-log.prob)
    }


    for (i in 1:length(ecs)) {
        ec <- pars$ecs[[i]]
        #print(ec)
                                        #print(c(fs[i], peak.time[i], spread[i]))

        tmp <- optimize(ob.func, interval=c(sum(ec),10^7),  ec=ec, peak.time=peak.time[i], spread=spread[i])

        fs[i] <- tmp$minimum
    }

    return(fs)

}


##' Function for doing random forrest fit of the K data appropriate for
##' passing in to epiInf.EM
##'
##' @param x the data frame to fit basedd on
##'
##' @return a fit random forrest model
##'
stat.mdl.linear.fit <- function(x) {
    require(rpart)

    f <- formula(paste0("K~",paste(names(subset(x,select=-K)),collapse="+")))
    rc <- lm(f, data=x)

    return(rc)
}



##'Function for doing the predict for duing the predict for the
##' results from a stat.mdl.rf.fit
##'
##' @param mdl the fit model
##'
##' @return a vector of predicted final sizes
##'
stat.mdl.linear.pred <- function(mdl) {
    rc <-predict(mdl)
    rc[rc<1] <- 1
    return(rc)
}



##' Function for doing superlearner fit of the K data appropriate for
##' passing in to iInf.EM
##'
##' @param x the data frame to fit basedd on
##'
##' @return a fit superlearner model
##'
stat.mdl.sl.fit <- function(x) {
  require(SuperLearner)
  require(gam)
  require(rpart)
  require(randomForest)
  y <- x$K
  x <- as.data.frame(subset(x, select=-K))
  rc <- SuperLearner(X = x, Y = y, newX = x, family = "gaussian", SL.library = c("SL.rpart", "SL.randomForest", "SL.glm", 'SL.gam')) #'SL.gam',
  ##rc <- SuperLearner(X = x, Y = y, newX = x, family = "gaussian", SL.library = c("SL.rpart", "SL.glm", 'SL.gam')) #'SL.gam',

  return(rc)
}
##'Function for doing the predict for duing the predict for the
##' results from a stat.mdl.sl.fit
##'
##' @param mdl the fit model
##'
##' @return a vector of predicted final sizes
##'
stat.mdl.sl.pred <- function(mdl) {
  return(pmax(1,predict(mdl, onlySL = T)$pred))
}
