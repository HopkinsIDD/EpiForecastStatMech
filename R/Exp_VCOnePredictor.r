source("R/EpiProgInf.r")
source("R/makebetas.R")


n.sims <-  500 # number of simulations
 n.epis <- 100 #number of simulated epidemics per simulation

 fs <- matrix(nrow=n.epis, ncol=n.sims) #matrix to hold final sizes
 pcts <- matrix(nrow=n.epis, ncol=n.sims) #matrix to hold proportion complete at time of simulation

 #initialize each model
 
 K.VC_linear <- fs
 K.VC_rpart <- fs
 K.VC_super <- fs
 K.norm_linear <-fs
 K.norm_rpart <-fs
 K.norm_super <- fs
 K.VC_alone <- fs
 K.norm_alone <- fs
 K.linear_alone <- fs
 K.rpart_alone <- fs
 K.super_alone <- fs

 for (i in 1:n.sims) {

   cat("iteration ", i, "\n")

  x <- runif(n.epis, 0, 3)
  fs[,i] <- rpois(n.epis, 1000*3^x)
  pcts[,i] <- runif(n.epis, .05,1)

  ecs <- gen.epicurves.ViboudChowell(pcts[,i], fs[,i], r = 1, p = 0.6)

  dat <- data.frame(K=2*sapply(ecs,sum),
                    x=x)

  K.VC_linear[,i] <- epiInf.EM(ecs, dat,
                               stat.mdl.linear.fit,
                               epi.mdl.ViboudChowell.fit2,
                               stat.mdl.linear.pred,
                               epi.mdl.ViboudChowell.pred2,
                               threshold = 1)$K

  K.VC_rpart[,i] <- epiInf.EM(ecs, dat,
                               stat.mdl.rpart.fit,
                               epi.mdl.ViboudChowell.fit2,
                               stat.mdl.rpart.pred,
                               epi.mdl.ViboudChowell.pred2,
                               threshold = 1)$K

   K.VC_super[,i] <- epiInf.EM(ecs, dat,
                               stat.mdl.sl.fit,
                               epi.mdl.ViboudChowell.fit2,
                               stat.mdl.sl.pred,
                               epi.mdl.ViboudChowell.pred2,
                               threshold = 1)$K


  K.norm_linear[,i] <- epiInf.EM(ecs, dat,
                               stat.mdl.linear.fit,
                               epi.mdl.normcurve.fit,
                               stat.mdl.linear.pred,
                               epi.mdl.normcurve.pred,
                               threshold = 1)$K

  K.norm_rpart[,i] <- epiInf.EM(ecs, dat,
                               stat.mdl.rpart.fit,
                               epi.mdl.normcurve.fit,
                               stat.mdl.rpart.pred,
                               epi.mdl.normcurve.pred,
                               threshold = 1)$K

   K.norm_super[,i] <- epiInf.EM(ecs, dat,
                               stat.mdl.sl.fit,
                               epi.mdl.normcurve.fit,
                               stat.mdl.sl.pred,
                               epi.mdl.normcurve.pred,
                               threshold = 1)$K

   K.VC_alone[,i] <- epi.mdl.ViboudChowell.init2(ecs)$K

   K.norm_alone[,i] <- epi.mdl.normcurve.init(ecs)$K


   tmp.dat <- dat
   tmp.dat$K <- sapply(ecs, sum)

   print("RPART STAT")
   K.rpart_alone[,i] <- stat.mdl.rpart.pred(stat.mdl.rpart.fit(tmp.dat))

   print("RPART LIN")
   K.linear_alone[,i] <- stat.mdl.linear.pred(stat.mdl.linear.fit(tmp.dat))

   print("RPART SUPER")
   K.super_alone[,i] <- stat.mdl.sl.pred(stat.mdl.sl.fit(tmp.dat))
 }


save(pcts, fs, K.VC_linear, K.VC_rpart, K.VC_super, K.norm_linear, K.norm_rpart, K.norm_super,
     K.VC_alone, K.norm_alone, K.rpart_alone, K.linear_alone, K.super_alone, file="VCSingleVarRuns100.RData")
