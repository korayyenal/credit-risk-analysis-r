# create necessary functions
createDMatrix <- function(x) {
  residential <- model.matrix(~TYP_RES-1,x)
  employment <- model.matrix(~ST_EMPL-1,x)
  
  X_numeric <- x %>%
    select_if(is.numeric)
  X_numeric <- cbind(X_numeric, residential, employment)
  X_matrix <- data.matrix(X_numeric)
  return (X_matrix)
}

# best cutoff point
opt.cut = function(perf, pred)
{
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(True_positive_rate = y[[ind]], False_positive_rate = x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
  return(cut.ind)
}

factorToNumeric <- function(y) {
  y_numeric = as.numeric(y)
  y_numeric[y_numeric==1] <- 0
  y_numeric[y_numeric==2] <- 1
  return(y_numeric)
}

dfToMatrixX = function(x)
{
  dummy_x <- model.matrix(~ ., data = x)
  df_x <- data.frame(dummy_x[,-1])
  matrix_x = as.matrix(select(df_x, -DEFAULTX1)[,])
  return(matrix_x)
}

dfToY = function(y)
{
  dummy_y <- model.matrix(~ ., data = y)
  df_y <- data.frame(dummy_y[,-1])
  dff_y = df_y[, "DEFAULTX1"]
  return(dff_y)
}

fnRate = function(predd, x, c=0.5)
{
  cutoff=c
  test.pred_lasso = rep(0, nrow(x))
  test.pred_lasso[predd > cutoff] = 1
  
  M_lasso=table(test.pred_lasso, credit_test_Y,dnn=c("Prediction","Observation"))
  a=M_lasso[1,1]
  b=M_lasso[1,2]
  c=M_lasso[2,1]
  d=M_lasso[2,2]
  
  fnr = b/(b+d)
  return(fnr)
}

fnRateXgb = function(predd, x, obs, c=0.5)
{
  cutoff=c
  test.pred_lasso = rep(0, nrow(x))
  test.pred_lasso[predd > cutoff] = 1
  
  M_lasso=table(test.pred_lasso, obs, dnn=c("Prediction","Observation"))
  a=M_lasso[1,1]
  b=M_lasso[1,2]
  c=M_lasso[2,1]
  d=M_lasso[2,2]
  
  fnr = b/(b+d)
  return(fnr)
}

fnRateAda = function(predd)
{
  M_ada <- predd$confusion
  a=M_ada[1,1]
  b=M_ada[1,2]
  c=M_ada[2,1]
  d=M_ada[2,2]

  fnr = b/(b+d)
  return(fnr)
}

f1Score = function(predd, x, c=0.5)
{
  cutoff=c
  test.pred_lasso = rep(0, nrow(x))
  test.pred_lasso[predd > cutoff] = 1
  
  M_lasso=table(test.pred_lasso, credit_test_Y,dnn=c("Prediction","Observation"))
  f1 <- confusionMatrix(M_lasso)$byClass['F1']
  return(f1)
}

f1ScoreXgb = function(predd, x, obs, c=0.5)
{
  cutoff=c
  test.pred_lasso = rep(0, nrow(x))
  test.pred_lasso[predd > cutoff] = 1
  
  M_lasso=table(test.pred_lasso, obs, dnn=c("Prediction","Observation"))
  f1 <- confusionMatrix(M_lasso)$byClass['F1']
  return(f1)
}

f1ScoreAda = function(predd)
{
  M_ada <- predd$confusion
  f1 <- confusionMatrix(M_ada)$byClass['F1']
  return(f1)
}

rocCurve <- function(predd, y) {
  pred=prediction(predd,y)
  perf=performance(pred,measure="tpr",x.measure="fpr")
  plot(perf)
  abline(a=0,b=1)
  optcut <- opt.cut(perf, pred)
  return(optcut)
}

dfToDMatrix <- function(df)
{
  y = df$DEFAULT
  x = df %>% select(-DEFAULT)
  y_numeric <- factorToNumeric(y)
  x_matrix <- createDMatrix(x)
  dmatrix <- xgb.DMatrix(data = x_matrix, label= y_numeric)
  return(dmatrix)
}

dfToYNumeric <- function(df)
{
  y = df$DEFAULT
  x = df %>% select(-DEFAULT)
  y_numeric <- factorToNumeric(y)
  return(y_numeric)
}

dfToDMatrixRegr <- function(df)
{
  df_regr <- df
  df_regr$DEFAULT <- NULL
  y_regr = df_regr$MNT_DEMANDE
  x_regr <- df_regr %>% select(-MNT_DEMANDE)
  x_matrix_regr <- createDMatrix(x_regr)
  dmatrix_regr <- xgb.DMatrix(data = x_matrix_regr, label= y_regr)
  return(dmatrix_regr)
}