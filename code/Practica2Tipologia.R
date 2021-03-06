#Instalamos los paquetes y las librer�as.

install.packages("dplyr")
install.packages("bindrcpp")
install.packages("splines")
install.packages("foreach")
install.packages("sqldf")
install.packages("graphicsQC")
install.packages("boxplotdbl")
install.packages("VIM")
install.packages("goftest")
install.packages("nortest")

library("nortest")
library(boxplotdbl)
library(sqldf)
library(gam)
library(dplyr)

#Leemos los datos 
getwd()
data=read.table("../data/datosTitanic.txt",header=TRUE, sep=",",na.strings = "")
head(data)
dim.data.frame(data)
names(data)

#Vemos el tipo de dato de cada variable. 
sapply(data, function(x) class(x))


#Vemos si hay elementos duplicados. Con el comando unique() eliminamos las muestras duplicadas  y vemos que sigue teniendo
#891 muestras. Por tanto, no hay duplicados. 
datasindup<-unique(data)
dim.data.frame(datasindup)

#Eliminamos dos variables "Passengerld","Name" y "Ticket"
data<-select(data,-PassengerId,-Name,-Ticket)

#Reemplazamos los valores perdidos en 'Age'con la media del resto de los no-missings.
#data<-na.gam.replace(data)

#Calculo la media y reemplazo los valores perdidos en 'Age' con la edad media redondeada.

media= mean(data$Age, na.rm = TRUE)
data$Age<- replace(data$Age, is.na(data$Age), ceiling(media))
media


# Imputaci�n de valores mediante la funci�n kNN() del paquete VIM

suppressWarnings(suppressMessages(library(VIM)))
data$Age <- kNN(data)$Age
data$Cabin<-kNN(data)$Cabin
data$Embarked<-kNN(data)$Embarked


#Veremos el n�mero de valores err�neos o desconocidos por atributo
sapply(data,function(x) sum(is.na(x)))

#Eliminar los registros con valores err�neos.
datasinNA<- na.omit(data)
dim(datasinNA)


sqldf("select survived, count(*) from data
          where age is null
          group by survived")
sqldf("select count(*) from data
          where survived=0")


#Vemos los outliers. 

#Mostramos los gr�ficos y despu�s obtenemos los valores de estos datos at�picos con la funci�n boxplot.stats()
#EDAD
boxplot(data$Age~data$Pclass,xlab="Pclass",ylab="Age",col=c("blue","yellow","red"))
boxplot.stats(data$Age)$out

#SibSp
boxplot(data$SibSp~data$Pclass,xlab="Pclass",ylab="SibSp",col=c("blue","yellow","red"))
boxplot.stats(data$SibSp)$out

#Fare
boxplot(data$Fare~data$Pclass,xlab="Pclass",ylab="Fare",col=c("blue","yellow","red"))
boxplot.stats(data$Fare)$out


#Exportaci�n de los datos limpios en .csv 
write.csv(data, "../data/datos_titanic_limpieza.csv")
  

####################-----------------AN�LISIS---------------------------############################


# Agrupaci�n por clase
pasajeros1clase<-data[data$Pclass==1,]
pasajeros2clase <- data[data$Pclass==2,]
pasajeros3clase <- data[data$Pclass==3,]

# Agrupaci�n por g�nero
Hombres<-data[data$Sex=='male',]
Mujeres<- data[data$Sex=='female',]

#Agrupaci�n por puerto en el que embarcaron.
PuertoC<-data[data$Embarked=='C',]
PuertoQ <- data[data$Embarked=='Q',]
PuertoS <- data[data$embarked=='S',]

# Histograma y curva normal sobreimpuesta 
x <- data$Age
h <- hist(x, breaks = 10, xlab = "Edad", main = "Histogram con curva normal")
xfit <- seq(min(x), max(x), length = 20)
yfit <- dnorm(xfit, mean = mean(x), sd = sd(x))
yfit <- yfit * diff(h$mids[1:2]) * length(x)
lines(xfit, yfit, col = "blue", lwd = 2)



require(MASS)
ajuste <- fitdistr(data$Age,"normal")
ajuste

#Aplicamos el test de Kolmorow-Smirnov

testKs<- ks.test(data$Age, "pnorm", mean =ajuste$estimate[1], sd= ajuste$estimate[2])
testKs

# Anderson-Darling test para la edad

testAd<-ad.test(data$Age)
testAd

#Vamos a verlo en un gr�fico para confirmarlo.

xe <- seq(min(data$Age), max(data$Age), by=5)
plot(xe, pnorm(xe, mean=ajuste$estimate[1], sd=ajuste$estimate[2]), type="l", col="red", xlab="edad", ylab="pnorm(x, mean, sd)")
plot(ecdf(data$Age), add=TRUE)


#Aplicamos el test de Anderson-Darling test para todas las variables.

library(nortest)
alpha = 0.05
col.names = colnames(data)
for (i in 1:ncol(data)) 
{ if (i == 1) cat("Variables que no siguen una distribuci�n normal:\n") 
  if (is.integer(data[,i]) | is.numeric(data[,i])) {
    p_val = ad.test(data[,i])$p.value 
    if (p_val < alpha) { 
      cat(col.names[i])
  # Format output 
      if (i < ncol(data) - 1) cat(", ") 
      if (i %% 3 == 0) cat("\n")
    }
  }
}


#Aplicamos el test de Fligner-Killeen para contrastar la homogeneidad de las varianzas
# de los diferentes grupos seg�ne l puerto donde embarcaron.

fligner.test(data$Survived ~ data$Embarked, data= data)

#Definimos las variables
Survived<-data$Survived
Pclass<-data$Pclass
Sexo<-data$Sex
SibSp<-data$SibSp
Parch<-data$Parch
Fare<-data$Fare 
Cabin<-data$Cabin
Embarked<-data$Embarked
Age<-data$Age

#Modelo de la regresi�n log�stica.

mod_log=glm(Survived~Pclass+Sexo+SibSp+Parch+Fare+Age,data=data,family=binomial(link="logit"))
summary(mod_log)

#Modelo de la regresi�n log�stica simplificada.
mod_log_simplif=glm(Survived~Pclass+Sexo+SibSp+Age,data=data,family=binomial(link="logit"))
summary(mod_log_simplif)


#Aplicamos el ANOVA para comparar los modelos.
anova(mod_log_simplif,mod_log)

#calculo del p-valor.
1-pchisq(0.98786,2)

#Aplicamos el m�todo step para la eliminaci�n de las variables no significativas. 
step(mod_log)

#Leemos los datos del test para hacer las predicciones.
datos_test<-read.table("../data/datos_test.txt",header = TRUE,sep=",",na.strings = "")

#Sustituimos los datos err�neos mediante el m�todo de los k-vecinos m�s pr�ximos. 
datos_test$Age <- kNN(datos_test)$Age
datos_test$Cabin<-kNN(datos_test)$Cabin
datos_test$Fare<-kNN(datos_test)$Fare
sapply(data, function(x) sum(is.na(x)))


#Definimos las variables de este nuevo dataset.
Survived<-datos_test$Survived
Pclass<-datos_test$Pclass
Sexo<-datos_test$Sex
SibSp<-datos_test$SibSp
Parch<-datos_test$Parch
Fare<-datos_test$Fare 
Cabin<-datos_test$Cabin
Embarked<-datos_test$Embarked
Age<-datos_test$Age

#Realizamos la predicci�n la variable 'Survived' de mis datos de test con el modelo log�stico simplif.
prediccion<-round(predict(mod_log_simplif,newdata=datos_test,type="response"))
prediccion
#A�adimos la columna de la predicci�n.
datos_test["Survived"]<-prediccion

#Representaci�n de los datos en gr�ficos.

#Gr�fico de sobrevivientes para cada clase
barplot(table(data$Survived,data$Pclass),
        main = "Sobrevivientes para cada clase",
        xlab = "Clase",
        col = c("red","blue"),beside = TRUE)
legend("topleft",
       c("No sobrevivieron","Sobrevivieron"),
       fill = c("red","blue"))

#Gr�fico de sobrevivientes por g�nero
barplot(table(data$Survived,data$Sex),
        main = "Sobrevivientes por Sexo",
        xlab = "G�nero",
        col = c("red","blue"),beside = TRUE)
legend("topleft",
       c("No sobrevivieron","Sobrevivieron"),
       fill = c("red","blue"))

#Gr�fico de sobrevivientes seg�n el puerto en el que embarcaron

barplot(table(data$Survived,data$Embarked),
        main = "Sobrevivientes por Puerto de Embarque",
        xlab = "Puerto de Embarque",
        col = c("red","blue"),beside = TRUE)
legend("topleft",
       c("No sobrevivieron","Sobrevivieron"),
       fill = c("red","blue"))


#Gr�fico de sobrevivientes para cada clase de los datos del test.
barplot(table(datos_test$Survived,datos_test$Pclass),
        main = "Sobrevivientes de los datos del test para cada clase",
        xlab = "Clase",
        col = c("green","yellow"),beside = TRUE)
legend("topleft",
       c("No sobrevivieron","Sobrevivieron"),
       fill = c("green","yellow"))

#Gr�fico de sobrevivientes por g�nero
barplot(table(datos_test$Survived,datos_test$Sex),
        main = "Sobrevivientes de los datos del test por Sexo",
        xlab = "G�nero",
        col = c("green","yellow"),beside = TRUE)
legend("topleft",
       c("No sobrevivieron","Sobrevivieron"),
       fill = c("green","yellow"))

#Por �ltimo escribimos los datos en un csv con la columna de la predicci�n de 'Survived'.

write.csv(datos_test,"../data/datos_test_predicc.csv")
