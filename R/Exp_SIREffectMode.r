source("R/EpiProgInf.r")
source("R/makebetas.R")
n.sims <- 500
 n.epis <- 100

 fs <- matrix(nrow=n.epis, ncol=n.sims)
 pcts <- matrix(nrow=n.epis, ncol=n.sims)

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


  effect.mod.betas <- gen_betas_effect_mod(n.epis)

  N <- nrow(effect.mod.betas)
  pcts[,i] <- runif(N, .05,1)

  gamma <- 1/3
  beta <- effect.mod.betas$beta1/3

  pop_sz <- rep(10000, N)

  ecs <- gen.epidemic.curves.SIR(pcts[,i], beta,pop_sz, gamma)


  ## Need to decompose ecs into ecs and true final sizes
  fs[,i] <- ecs$fs
  ecs <- ecs$ecs


  ##########RUN THE EM ALGORITHM
  dat <- data.frame(K=0, hd=effect.mod.betas$hd,
                    ws = effect.mod.betas$ws)

  dat$K <- 2*sapply(ecs, sum)

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
     K.VC_alone, K.norm_alone, K.rpart_alone, K.linear_alone, K.super_alone, file="SIREffectModRuns100.RData")
