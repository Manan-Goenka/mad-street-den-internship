install.packages("RPostgreSQL")
install.packages("ggplot2")
install.packages("cluster")

library(RPostgreSQL)
# connection to redshift
connect_to_redshift <- function(){
  redshift_host <- "optimizedcluster.czunimvu3pdw.us-west-2.redshift.amazonaws.com"
  redshift_port <- "5439"
  redshift_user <- "manan"
  redshift_password <- "Manan123"
  redshift_db <- "msd"
  
  drv <- dbDriver("PostgreSQL")
  conn <- dbConnect(
    drv,
    host=redshift_host,
    port=redshift_port,
    user=redshift_user,
    password=redshift_password,
    dbname=redshift_db)
  return(conn)
}

query <- "SELECT
uuid,
pages_viewed_before_buy,
items_viewed_before_buy,
brands_viewed_before_buy,
categories_viewed_before_buy,
time_between_first_view_and_buy,
frequency_of_buys_category,
recent_buy_time,
number_of_colours_addtocart_or_buy_in_this_category,
number_of_styles_addtocart_or_buy_in_this_category,
sum_add_to_cart,
avg_add_to_cart,
sum__buy,
avg__buy,
central_dollarvalue_difference,
sessions_without_significant_event FROM user_cohorting where ontology='Female>Apparel>Womens Apparel>Womens Western Wear>Tops and tees';"
data = dbGetQuery(conn = connect_to_redshift(),query)

normalized_data= apply(X = data[,-which(names(data) == "uuid")],MARGIN = 2,FUN = function(x){
  (x - mean(x))/sd(x)
})

mean(normalized_data[,1])

normalized_data=data.frame(normalized_data)

Price_Cluster <- kmeans(normalized_data[, which(names(normalized_data) %in% c("sum_add_to_cart","avg_add_to_cart","sum__buy","avg__buy","central_dollarvalue_difference",
                                                                   "sessions_without_significant_event"))], 4, nstart = 20)

library(ggplot2)

Total_Cluster$cluster <- as.factor(Total_Cluster$cluster)
ggplot(normalized_data, aes(pages_viewed_before_buy, items_viewed_before_buy, color = Total_Cluster$cluster)) + geom_point()

pdf("Price_Cluster4.pdf")
with(Price_Cluster, pairs(normalized_data[, which(names(normalized_data) %in% c("sum_add_to_cart","avg_add_to_cart","sum__buy","avg__buy","central_dollarvalue_difference",
                                                                                "sessions_without_significant_event"))], col=c(1:6)[Price_Cluster$cluster]))
dev.off()

library(cluster)
pdf("Price_Cusplot5.pdf")
clusplot(normalized_data, Total_Cluster$cluster, color=TRUE, shade=TRUE, 
         labels=2, lines=0)
dev.off()

query <- "SELECT central_dollarvalue_difference, COUNT (*) FROM user_cohorting where ontology='Female>Apparel>Womens Apparel>Womens Western Wear>Tops and tees' GROUP BY 1;"
cdvd_data = dbGetQuery(conn = connect_to_redshift(),query)

hist(data[, 15], )

plot(cdvd, pch=16, col='black', cex=0.5)

d2=data[, data[, 2]>0]
d2=data.frame(d2)
d2=d2[,1][d2[, 1]>0]
d2=data.frame(d2)

hist(d2[,1], labels=T)

data2=data[data$pages_viewed_before_buy!=0, ]

normalized_data= apply(X = data2[,-which(names(data2) == "uuid")],MARGIN = 2,FUN = function(x){
  (x - mean(x))/sd(x)
})

wss <- sapply(1:10, function(k){kmeans(normalized_data[, which(names(normalized_data) %in% c("sum_add_to_cart","avg_add_to_cart","sum__buy","avg__buy","central_dollarvalue_difference",
                                                                                             "sessions_without_significant_event"))], k, nstart=20,iter.max = 15 )$tot.withinss})

plot(1:10, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


Total_Cluster <- kmeans(normalized_data, 7, nstart = 20)

pdf("Plot1.pdf")
plot(normalized_data[, 1], normalized_data[, 14], main="Scatterplot", 
     xlab="Pages_Viewed_Before_Buy", ylab="Central_Dollar-Value_Difference", pch=19)
dev.off()

res <- cor(normalized_data)
