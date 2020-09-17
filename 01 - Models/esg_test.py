
import math
import random
import os
import numpy as np

def instantaneousForwardRate(timeToMaturity, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3):
    #Renvoie le taux forward instantané de maturité timeTomaturity avec les paramètres spécifiés, dans le modèle NSS
    
    return beta_0 + beta_1 * math.exp(-timeToMaturity / lambda_1) + beta_2 * timeToMaturity / lambda_1 * math.exp(-timeToMaturity / lambda_1) + beta_3 * timeToMaturity / lambda_2 * math.exp(-timeToMaturity / lambda_2)

def functionPhiG2PP( time, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_a, coeff_b, sigma, eta, rho):
    #Renvoie le phi(t) du modèle G2++
    
    return instantaneousForwardRate(time, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3) + sigma *sigma / (2 * coeff_a *coeff_a) * (1 - math.exp(-coeff_a * time)) *(1 - math.exp(-coeff_a * time)) + eta *eta / (2 * coeff_b *coeff_b) * (1 - math.exp(-coeff_b * time)) *(1 - math.exp(-coeff_b * time)) + rho * eta * sigma / (coeff_a * coeff_b) * (1 - math.exp(-coeff_a * time)) * (1 - math.exp(-coeff_b * time))

def fCIR( time, kappa, theta, sigma, zVariable):
    #Renvoie le f CIR utilisé pour calculer le phi_t dans le modèle CIR2++
    
    h = math.sqrt(kappa * kappa + 2 * sigma *sigma)
    
    return (2 * kappa * theta * (math.exp(time * h) - 1)) / (2 * h + (kappa + h) * (math.exp(time * h) - 1)) + zVariable * (4 * h *h * math.exp(time * h)) / ((2 * h + (kappa + h) * (math.exp(time * h) - 1)) *(2 * h + (kappa + h) * (math.exp(time * h) - 1)))

def functionPhiCIR2PP( time, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, xVariable, kappa_2, theta_2, sigma_2, yVariable):
    #Renvoie la fonction varphi (permettant de calibrer la courbe de taux initial dans le modèle CIR2++
    
    forward = instantaneousForwardRate(time, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3)
    fCIR1 = fCIR(time, kappa_1, theta_1, sigma_1, xVariable)
    fCIR2 = fCIR(time, kappa_2, theta_2, sigma_2, yVariable)
    
    return forward - fCIR1 - fCIR2

def AfunctionCIR2PP( initialDate, maturityDate, kappa, theta, sigma):
    #Fonction A dans le modèle CIR2++, utilisée pour simplifier les notations
    
    h = math.sqrt(kappa*kappa + 2 * sigma *sigma)
    numerateurA = 2 * h * math.exp((kappa + h) * (maturityDate - initialDate) / 2)
    denominateurA = 2 * h + (kappa + h) * (math.exp((maturityDate - initialDate) * h) - 1)
    
    return math.exp(((2 * kappa * theta) / (sigma *sigma))*math.log(numerateurA / denominateurA))

def BfunctionCIR2PP( initialDate, maturityDate, kappa, theta, sigma):
    #Fonction B dans le modèle CIR2++, utilisée pour simplifier les notations
    
    h = math.sqrt(kappa*kappa + 2 * sigma *sigma)
    numerateurB = 2 * (math.exp((maturityDate - initialDate) * h) - 1)
    denominateurB = 2 * h + (kappa + h) * (math.exp((maturityDate - initialDate) * h) - 1)
    
    return (numerateurB / denominateurB)

def PfunctionCIR2PP( initialDate, maturityDate, kappa, theta, sigma, z):
    #Fonction P^CIR dans le modèle CIR2++, utilisée pour simplifier les notations
    
    return AfunctionCIR2PP(initialDate, maturityDate, kappa, theta, sigma) * math.exp(-BfunctionCIR2PP(initialDate, maturityDate, kappa, theta, sigma) * z)

def tauxNSS( timeToMaturity, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3):
    #Renvoie taux (composition continue) de maturité timeTomaturity avec les paramètres spécifiés, dans le modèle NSS
    if timeToMaturity == 0:
        return beta_0+beta_1
    else:
        return beta_0 + beta_1 * ((1 - math.exp(-timeToMaturity / lambda_1)) / (timeToMaturity / lambda_1)) + beta_2 * ((1 - math.exp(-timeToMaturity / lambda_1)) / (timeToMaturity / lambda_1) - (math.exp(-timeToMaturity / lambda_1))) + beta_3 * ((1 - math.exp(-timeToMaturity / lambda_2)) / (timeToMaturity / lambda_2) - (math.exp(-timeToMaturity / lambda_2)))

def marketInitialZeroCouponPrice( timeToMaturity, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3):
    #Renvoie le prix du Zero Coupon (composition continue) de maturité timeTomaturity avec les paramètres spécifiés, dans le modèle NSS
    return math.exp(-tauxNSS(timeToMaturity, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3) * timeToMaturity)

def zeroCouponPriceCIR2PP( initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, kappa_2, theta_2, sigma_2, x_0, y_0, x_t, y_t):
    #Pricing des Zero-Coupon, modèle CIR2++
    
    phiXsi = (marketInitialZeroCouponPrice(maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3) / marketInitialZeroCouponPrice(initialDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3)) * (PfunctionCIR2PP(0, initialDate, kappa_1, theta_1, sigma_1, x_0) / PfunctionCIR2PP(0, maturityDate, kappa_1, theta_1, sigma_1, x_0)) * (PfunctionCIR2PP(0, initialDate, kappa_2, theta_2, sigma_2, y_0) / PfunctionCIR2PP(0, maturityDate, kappa_2, theta_2, sigma_2, y_0))
    pXsi = PfunctionCIR2PP(initialDate, maturityDate, kappa_1, theta_1, sigma_1, x_t) * PfunctionCIR2PP(initialDate, maturityDate, kappa_2, theta_2, sigma_2, y_t)
    
    return phiXsi * pXsi

def zeroCouponYieldCIR2PP( initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, kappa_2, theta_2, sigma_2, x_0, y_0, x_t, y_t):
    #Fonction B dans le modèle CIR2++
    
    return -(math.log(zeroCouponPriceCIR2PP(initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, kappa_2, theta_2, sigma_2, x_0, y_0, x_t, y_t)) / (maturityDate - initialDate))

def forwardSwapRateCIR2PP( pricingDate, timeToMaturity, timeToTenor, deltaTime, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, kappa_2, theta_2, sigma_2, x_0, y_0, x_t, y_t):
    #Renvoie le taux swap forward, vu à la date pricingDate, 
    #pour un swap démarrant en date pricingDate + timeToMaturity, 
    #payant des flux à partir de la date suivante jusqu'à la date pricingDate + timeToMaturity + timeToTenor
    #les flux sont payés à la fréquence deltaTime (timeToTenor / deltaTime doit être un entier)
    numerateur = 0
    denominateur = 0
    
    numerateur = zeroCouponPriceCIR2PP(pricingDate, pricingDate + timeToMaturity, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, kappa_2, theta_2, sigma_2, x_0, y_0, x_t, y_t)
    numerateur = numerateur-zeroCouponPriceCIR2PP(pricingDate, pricingDate + timeToMaturity+timeToTenor, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, kappa_2, theta_2, sigma_2, x_0, y_0, x_t, y_t)
    
    for i in range(timeToTenor//deltaTime):
        denominateur += deltaTime*zeroCouponPriceCIR2PP(pricingDate, pricingDate + timeToMaturity + (i+1)*deltaTime, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1, theta_1, sigma_1, kappa_2, theta_2, sigma_2, x_0, y_0, x_t, y_t)
        
    return numerateur / denominateur

def vFunctionG2PP( initialDate, maturityDate, coeff_a, sigma, coeff_b, eta, rho):
    #Fonction V dans le modèle G2++, utilisée pour simplifier les notations
    
    return ((sigma / coeff_a) *(sigma / coeff_a)) * (maturityDate - initialDate + ((2 / coeff_a) * math.exp(-coeff_a * (maturityDate - initialDate))) - ((1 / (2 * coeff_a)) * math.exp(-2 * coeff_a * (maturityDate - initialDate))) - (3 / (2 * coeff_a))) + ((eta / coeff_b) *(eta / coeff_b)) * (maturityDate - initialDate + ((2 / coeff_b) * math.exp(-coeff_b * (maturityDate - initialDate))) - ((1 / (2 * coeff_b)) * math.exp(-2 * coeff_b * (maturityDate - initialDate))) - (3 / (2 * coeff_b))) + ((2 * rho * sigma * eta) / (coeff_a * coeff_b)) * (maturityDate - initialDate + ((math.exp(-coeff_a * (maturityDate - initialDate)) - 1) / (coeff_a)) + ((math.exp(-coeff_b * (maturityDate - initialDate)) - 1) / (coeff_b)) - ((math.exp(-(coeff_a + coeff_b) * (maturityDate - initialDate)) - 1) / (coeff_a + coeff_b)))

def AfunctionG2PP( initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_a, sigma, coeff_b, eta, rho):
    #Fonction A dans le modèle G2++, utilisée pour simplifier les notations
    
    return (marketInitialZeroCouponPrice(maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3) / marketInitialZeroCouponPrice(initialDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3)) * (math.exp((1 / 2) * (vFunctionG2PP(initialDate, maturityDate, coeff_a, sigma, coeff_b, eta, rho) - vFunctionG2PP(0, maturityDate, coeff_a, sigma, coeff_b, eta, rho) + vFunctionG2PP(0, initialDate, coeff_a, sigma, coeff_b, eta, rho)))) 

def BfunctionG2PP( zReal, initialDate, maturityDate):
    #Fonction B dans le modèle G2++, utilisée pour simplifier les notations
    
    return (1 - exp(-zReal * (maturityDate - initialDate))) / zReal

def zeroCouponPriceG2PP( initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_a, sigma, coeff_b, eta, rho, xFactor, yFactor):
    #Princing des Zero-Coupon, modèle G2++
    
    return AfunctionG2PP(initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_a, sigma, coeff_b, eta, rho) * math.exp(-BfunctionG2PP(coeff_a, initialDate, maturityDate) * (xFactor)- BfunctionG2PP(coeff_b, initialDate, maturityDate) * (yFactor))

def zeroCouponYieldG2PP( initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_a, sigma, coeff_b, eta, rho, xFactor, yFactor):
    #Renvoie le taux Zero Coupon de maturité maturityDate à la date initialDate
    
    return -(math.log(zeroCouponPriceG2PP(initialDate, maturityDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_a, sigma, coeff_b, eta, rho, xFactor, yFactor)) / (maturityDate - initialDate))

def generateur_normale( p):
    
    a1 = -39.69683028665376
    a2 = 220.9460984245205
    a3 = -275.9285104469687
    a4 = 138.3577518672690
    a5 =-30.66479806614716
    a6 = 2.506628277459239
    
    b1 = -54.47609879822406
    b2 = 161.5858368580409
    b3 = -155.6989798598866
    b4 = 66.80131188771972
    b5 = -13.28068155288572
    
    c1 = -0.007784894002430293
    c2 = -0.3223964580411365
    c3 = -2.400758277161838
    c4 = -2.549732539343734
    c5 = 4.374664141464968
    c6 = 2.938163982698783
    
    d1 = 0.007784695709041462
    d2 = 0.3224671290700398
    d3 = 2.445134137142996
    d4 = 3.754408661907416
    
    #Define break-points.
    
    p_low =  0.02425;
    p_high = 1 - p_low
    sigma = 1.318
    my = 9.357
    
    #Rational approximation for lower region.
    
    if (0 < p and p < p_low): 
        q = math.sqrt(-2*math.log(p))
        x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    
    #Rational approximation for central region.
    if (p_low <= p and p <= p_high):
        q = p - 0.5
        r = q*q
        x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
        
    #Rational approximation for upper region.
    if (p_high < p and p < 1):
        q = math.sqrt(-2*math.log(1-p))
        x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    
    return x

def payoffMaxZero(x):
    
    if x>0:
        return x
    else:
        return 0

#Debut du code pricerAsiatique

import time

start = time.perf_counter()
temp_start = start

#Definition de la variable pour reinitialiser le generateur d'uniforme, utiliser l'un ou l'autre des deux, mais pas les deux en meme temps
#1) Initialisation a partir de l'horloge (retirer le "#" )
#import datetime

#init_alea=datetime.datetime.now
#idum=-inti-ale

#2) Initialisation fixe pour generer les memes scenarios
idum = -1


import os

#Ouverture du fichier
nom_du_fichier='Projection - Inputs.txt'
fichier=open(nom_du_fichier,'r')
contenu_fichier=fichier.read()
#Passage du contenu de chaine de caractere a liste
contenu_liste=contenu_fichier.split('\n') #separateur des lignes
for i in range(len(contenu_liste)):
    contenu_liste[i]=contenu_liste[i].split(',') #separateur des colonnes

fichier.close() #ferme le fichier

if contenu_liste[0][0]=='"CIR2++"':
    modelCIR2PP = True
    modelG2PP = False
else:
    modelCIR2PP = False
    modelG2PP = True

if modelCIR2PP == True:
    kappa_1CIR2PP = float(contenu_liste[1][0])
    theta_1CIR2PP = float(contenu_liste[2][0])
    sigma_1CIR2PP = float(contenu_liste[3][0])
    kappa_2CIR2PP = float(contenu_liste[4][0])
    theta_2CIR2PP = float(contenu_liste[5][0])
    sigma_2CIR2PP = float(contenu_liste[6][0])
    x_0CIR2PP = float(contenu_liste[7][0])
    y_0CIR2PP = float(contenu_liste[8][0])

if modelG2PP == True:
    coeff_aG2PP = float(contenu_liste[1][0])
    coeff_bG2PP = float(contenu_liste[2][0])
    sigmaG2PP = float(contenu_liste[3][0])
    eta = float(contenu_liste[4][0])
    roh2PP = float(contenu_liste[5][0])
    x_0G2PP = float(contenu_liste[6][0])
    y_0G2PP = float(contenu_liste[7][0])

#Parametre NSS
beta_0 = float(contenu_liste[11][0])
beta_1 = float(contenu_liste[12][0])
beta_2 = float(contenu_liste[13][0])
beta_3 = float(contenu_liste[14][0])
lambda_1 = float(contenu_liste[15][0])
lambda_2 = float(contenu_liste[16][0])

#Parametre Modele
anneeProjection = int(contenu_liste[17][0])#Nombre d'annees dans les projections

nombreSimuAnnee = int(contenu_liste[18][0])#Nombre de pas de temps dans chaque annee

splitAnnuelOutput = int(contenu_liste[19][0])#splitAnnuel denote le decoupe d'une annee qui est fait pour les sorties, exemple 2 : semestre et 4: trimestre, 250: jour
#pour info, si je veux prendre 4 par exemple, il faudra que je mette 256 

nombreSimu = int(contenu_liste[20][0])#Nombre de scenarios generes

payerIndex = int(contenu_liste[21][0])#PayerIndex=1 pour une swaption payeuse, 0 pour une receveuse

deltaTimeSwaption = int(contenu_liste[22][0])

MaturitySwaptionMax = int(contenu_liste[23][0])

splitAnnuelMaturitySwaption = float(contenu_liste[24][0]) #double correspondant à la fréquence d'évaluation des maturités des swaptions

nombreMaturity = int(contenu_liste[25][0])

tenorMax = int(contenu_liste[26][0])#Tenor maximal, toutes les maturites de 1 a cette valeur vont etre pricées

moneyNessIndex = int(contenu_liste[27][0])#permet de pricer des money-ness differents. Une valeur de 21 permet d'aller de 50% a 150% de 5% en 5%

#SORTIES
if contenu_liste[28][0]== '"True"':
    sortieBrowniens = True
else :
    sortieBrowniens = False
if contenu_liste[29][0]== '"True"':
    sortieScenarioDeflateur = True
else :
    sortieScenarioDeflateur = False
if contenu_liste[30][0]== '"True"':
    sortieScenarioTaux = True
else :
    sortieScenarioTaux = False
if contenu_liste[31][0]== '"True"':
    sortieScenarioPrix = True
else :
    sortieScenarioPrix = False
if contenu_liste[32][0]== '"True"':
    sortieScenarioPrixZeroCouponUneMaturite = True
else :
    sortieScenarioPrixZeroCouponUneMaturite = False
if contenu_liste[33][0]== '"True"':
    sortieTestMartingaleDeflateur = True
else :
    sortieTestMartingaleDeflateur = False
if contenu_liste[34][0]== '"True"':
    sortieTestDistributionFacteurR = True
else :
    sortieTestDistributionFacteurR = False
if contenu_liste[35][0]== '"True"':
    sortieTestMartingaleMaturiteFixe = True
else :
    sortieTestMartingaleMaturiteFixe = False
    
if modelCIR2PP == True:
    if contenu_liste[36][0]== '"True"':
        pricerSwaptionCIR2PP = True
        pricerSwaptionG2PP = False
    else :
        pricerSwaptionCIR2PP = False
        pricerSwaptionG2PP = True

#BACKTEST
if contenu_liste[37][0]== '"True"':
    backtestNSS = True
else :
    backtestNSS = False
if contenu_liste[38][0]== '"True"':
    backtestSemestrielFacteurX = True
else :
    backtestSemestrielFacteurX = False
if contenu_liste[39][0]== '"True"':
    backtestSemestrielFacteurY = True
else :
    backtestSemestrielFacteurY = False
if contenu_liste[40][0]== '"True"':
    backtestSemestrielFacteurR = True
else :
    backtestSemestrielFacteurR = False
if contenu_liste[41][0]== '"True"':
    backtestSemestrielFacteurDeflateur = True
else :
    backtestSemestrielFacteurDeflateur = False
if contenu_liste[42][0]== '"True"':
    backtestPhiG2PP = True
else :
    backtestPhiG2PP = False

mature=[]
for i in range(43,len(contenu_liste)-1):
    mature.append(float(contenu_liste[i][0]))
initialMaturityVecteur = np.array(mature)

nombreTimeStep = nombreSimuAnnee*anneeProjection #Nombre de simulations temporelles dans chaque scenario
timeStep = 1.0/nombreSimuAnnee
timeIndex = 0.0
idum = -1

nombreTimeStepOutput= anneeProjection*splitAnnuelOutput

forwardSwapRate = 0
strike = 0
discountFactorSwaption = 0

if sortieBrowniens == True:
    ofBrownien1 = open("ofBrownien1.txt",'w')
    ofBrownien2Uncorrelated = open("ofBrownien2Uncorrelated.txt",'w')
    ofBrownien2Correlated = open("ofBrownien2Correlated.txt",'w')

#Sorties des tables G2++
if modelG2PP == True :
    ofSortieDeflateurG2PP = open("ofSortieDeflateurG2PP.txt",'w')

    ofSortieTauxG2PP = open("ofSortieTauxG2PP.txt",'w')

    ofSortieDiscountedPrixZeroCouponsG2PP = open("ofSortieDiscountedPrixZeroCouponsG2PP.txt",'w')

    ofMartingaleTestDiscZCPriceG2PP = open("ofMartingaleTestDiscZCPriceG2PP.txt",'w')

    ofPricerSwaptionG2PP = open("ofPricerSwaptionG2PP.txt",'w')

#Sorties de tables CIR2++
if modelCIR2PP == True :
    ofSortieDeflateurCIR2PP = open("ofSortieDeflateurCIR2PP.txt",'w')

    ofSortieTauxCIR2PP = open("ofSortieTauxCIR2PP.txt",'w')

    ofSortieDiscountedPrixZeroCouponsCIR2PP = open("ofSortieDiscountedPrixZeroCouponsCIR2PP.txt",'w')

    ofMartingaleTestDiscZCPriceCIR2PP = open("ofMartingaleTestDiscZCPriceCIR2PP.txt",'w')

    ofForwardSwapRateCIR2PP = open("ofForwardSwapRateCIR2PP.txt",'w')

    ofPricerSwaptionCIR2PP = open("ofPricerSwaptionCIR2PP.txt",'w')

if sortieTestMartingaleDeflateur == True:
    #Tableau qui va servir à faire le backtest du caractère martingale du déflateur dans le modèle G2++
    if modelG2PP == True:
        martingaleDiscountG2PP = np.zeros((4,nombreTimeStepOutput))

    #Tableau qui va servir à faire le backtest du caractère martingale du déflateur dans le modèle CIR2++
    if modelCIR2PP == True:
        martingaleDiscountCIR2PP = np.zeros((4,nombreTimeStepOutput))

if sortieTestDistributionFacteurR == True:
    rFactorDistribution = np.zeros((4,nombreTimeStepOutput))

#Tableau qui va servir à faire les calculs sur les propriétés martingales des prix actualisés des ZC dans le modèle G2++
if modelG2PP == True:
    martingaleTestTempG2PP = np.zeros((nombreMaturity,nombreTimeStepOutput+1))

#Tableau qui va servir à faire les calculs sur les propriétés martingales des prix actualisés des ZC dans le modèle CIR2++
if modelCIR2PP == True:
    martingaleTestTempCIR2PP = np.zeros((nombreMaturity,nombreTimeStepOutput+1))

#Sorties ecran
#Divers
print(nombreSimu,"\tScenarios:") 
print(anneeProjection,"\tyears simulated")
print(nombreTimeStep, "\ttimes steps generated")
print(nombreTimeStepOutput, "\ttime steps outputs")
print(nombreMaturity,"\tmaturities for each scenarios and outputs time steps\n")
#Paramètres
print("Nelson-Siegel-Svensson Model Parameters:")
print("\tlambda_1:\t", lambda_1)
print("\tlambda_2:\t", lambda_2)
print("\tbeta_0:\t", beta_0)
print("\tbeta_1:\t", beta_1)
print("\tbeta_2:\t", beta_2)
print("\tbeta_3:\t", beta_3)

if modelG2PP == True :
    print("G2++ Model Parameters:")
    print("\tcoefficient a:\t", coeff_aG2PP)
    print("\tcoefficient b:\t", coeff_bG2PP)
    print("\tsigmaG2PP:\t", sigmaG2PP)
    print("\teta:\t", eta)
    print("\trhoG2PP:\t", rhoG2PP)
    print("\tx_0G2PP:\t", x_0G2PP)
    print("\ty_0G2PP:\t", y_0G2PP)
    
if modelCIR2PP == True:
    print("CIR2++ Model Parameters:")
    print("\tx_0:\t", x_0CIR2PP)
    print("\tkappa_1:\t", kappa_1CIR2PP)
    print("\ttheta_1:\t", theta_1CIR2PP)
    print("\tsigma_1:\t", sigma_1CIR2PP)
    print("\ty_0:\t", y_0CIR2PP)
    print("\tkappa_2:\t", kappa_2CIR2PP)
    print("\ttheta_2:\t", theta_2CIR2PP)
    print("\tsigma_2:\t", sigma_2CIR2PP)

#Initialisation des tables pour les normales et les Browniens (tableau non vide en Python avec des zeros)
normaleBrownien1 = np.zeros((2,nombreTimeStep))
normaleBrownien2Uncorrelated = np.zeros((2,nombreTimeStep))
normaleBrownien2Correlated = np.zeros((2,nombreTimeStep))

if modelG2PP == True:
    #Initialisation des tables en frequence journaliere pour approximer l'integrale stochasique, model G2++
    xFactorG2PP = np.zeros((2,nombreTimeStep))
    yFactorG2PP = np.zeros((2,nombreTimeStep))
    rFactorG2PP = np.zeros((2,nombreTimeStep))
    deflatorG2PP = np.zeros((2,nombreTimeStep))
    #Initialisation des tables en fréquence correspondant à splitAnnuelOutput pour les sorties des facteurs, modèle G2++
    xFactorG2PPOutput = np.zeros((2,nombreTimeStep))
    yFactorG2PPOutput = np.zeros((2,nombreTimeStep))
    rFactorG2PPOutput = np.zeros((2,nombreTimeStep))
    deflatorG2PPOutput = np.zeros((2,nombreTimeStep))
    #Tables d'outputs de prix et taux ZC, modèle G2++
    #Initialisation des tables d'outputs des Taux Zero Coupons
    #Les taux sont en time to maturity fixes
    zeroCouponYieldOutputG2PP = np.zeros((2,nombreMaturity,nombreTimeStepOutput + 1))
    #la première dimension contient une simulation i donnée (centrale ou antithétique)
    #la deuxième dimension contient un timeToMaturity k+1 donné
    #La dernière dimension contient le timeStep j. Il y en a une de plus que de pas de temps simulés pour la date initiale

    #Initialisation des tables d'outputs des Prix Zero Coupons
    #Les prix sont en maturités fixes
    zeroCouponPriceOutputG2PP = np.zeros((2,nombreMaturity,nombreTimeStepOutput + 1))
    #la premiere dimension contient une simulation i donnee
    #la deuxieme dimension contient un timeToMaturity k+1 donne
    #la derniere dimension contient le timeStep j. Il y en a une de plus que de pas de temps simules pour la date initiale

    #Initialisation des tables d'outputs des prix Monte Carlo des swaptions
    pricerSwaptionOutputG2PP = np.zeros((nombreTimeStepOutput,tenorMax*deltaTimeSwaption,moneyNessIndex))

    
if modelCIR2PP == True:
    #Initialisation des tables en fréquence journalière pour approximer l'intégrale stochastique, modèle CIR2++
    xFactorCIR2PP = np.zeros((2,nombreTimeStep))
    yFactorCIR2PP = np.zeros((2,nombreTimeStep))
    rFactorCIR2PP = np.zeros((2,nombreTimeStep))
    deflatorCIR2PP = np.zeros((2,nombreTimeStep))
    #Initialisation des tables en fréquence correspondant à splitAnnuelOutput pour les sorties des facteurs, modèle CIR2++
    xFactorCIR2PPOutput = np.zeros((2,nombreTimeStep))
    yFactorCIR2PPOutput = np.zeros((2,nombreTimeStep))
    rFactorCIR2PPOutput = np.zeros((2,nombreTimeStep))
    deflatorCIR2PPOutput = np.zeros((2,nombreTimeStep))
    
    #Tables d'outputs de prix et taux ZC, modèle CIR2++
    #Initialisation des tables d'outputs des Taux Zero Coupons
    #Les taux sont en time to maturity fixes
    zeroCouponYieldOutputCIR2PP = np.zeros((2,nombreMaturity,nombreTimeStepOutput + 1))
    #la première dimension contient une simulation i donnée (centrale ou antithétique)
    #la deuxième dimension contient un timeToMaturity k+1 donné
    #La dernière dimension contient le timeStep j. Il y en a une de plus que de pas de temps simulés pour la date initiale

    #Initialisation des tables d'outputs des Prix Zero Coupons
    #Les prix sont en maturités fixes
    zeroCouponPriceOutputCIR2PP = np.zeros((2,nombreMaturity,nombreTimeStepOutput + 1))
    #la premiere dimension contient une simulation i donnee
    #la deuxieme dimension contient un timeToMaturity k+1 donne
    #la derniere dimension contient le timeStep j. Il y en a une de plus que de pas de temps simules pour la date initiale

    #Initialisation des tables d'outputs des prix Monte Carlo des swaptions
    pricerSwaptionOutputCIR2PP = np.zeros((nombreTimeStepOutput,tenorMax//deltaTimeSwaption,moneyNessIndex))

if modelG2PP == True:
    r_0G2PP = x_0G2PP + y_0G2PP + functionPhiG2PP(0, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, coeff_bG2PP, sigmaG2PP, eta, rhoG2PP)
if modelCIR2PP == True:
    r_0CIR2PP = x_0CIR2PP + y_0CIR2PP + functionPhiCIR2PP(0, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, x_0CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, y_0CIR2PP)

#Initialisation des taux swap forward
forwardSwapRateMatrix = np.zeros((MaturitySwaptionMax +1,tenorMax//deltaTimeSwaption))
for i in range(tenorMax//deltaTimeSwaption):
    for j in range(MaturitySwaptionMax+1):
        forwardSwapRateMatrix[j][i] = forwardSwapRateCIR2PP(0, j, (i + 1) *deltaTimeSwaption, deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, x_0CIR2PP, y_0CIR2PP)

#a dupliquer sur le G2PP
if (pricerSwaptionCIR2PP == True):
    #ofForwardSwapRateCIR2PP.write("Nelson-Siegel-Svensson Model Parameters;\n")
    #ofForwardSwapRateCIR2PP.write("\tlambda_1:\t" + str(lambda_1) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tlambda_2:\t" +str(lambda_2) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tbeta_0:\t" + str(beta_0) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tbeta_1:\t" + str(beta_1) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tbeta_2:\t" + str(beta_2) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tbeta_3:\t" + str(beta_3) + "\n")
    #ofForwardSwapRateCIR2PP.write("CIR2++ Model Parameters:\n")
    #ofForwardSwapRateCIR2PP.write("\tx_0:\t" + str(x_0CIR2PP) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tkappa_1:\t" + str(kappa_1CIR2PP) + "\n")
    #ofForwardSwapRateCIR2PP.write("\ttheta_1:\t" + str(theta_1CIR2PP) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tsigma_1:\t" + str(sigma_1CIR2PP) + "\n")
    #ofForwardSwapRateCIR2PP.write("\ty_0:\t" + str(y_0CIR2PP) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tkappa_2:\t" + str(kappa_2CIR2PP) + "\n")
    #ofForwardSwapRateCIR2PP.write("\ttheta_2:\t" + str(theta_2CIR2PP) + "\n")
    #ofForwardSwapRateCIR2PP.write("\tsigma_2:\t" + sigma_2CIR2PP + "\n";)
    ofForwardSwapRateCIR2PP.write("T=\t")
    for j in range(MaturitySwaptionMax+1):
        ofForwardSwapRateCIR2PP.write(str(j) + "\t")
    ofForwardSwapRateCIR2PP.write("\n")
    for i in range(tenorMax // deltaTimeSwaption): #on donne le taux swap que pour des maturités entieres, peut etre modifie facilement
        ofForwardSwapRateCIR2PP.write("T+" + str((i+1)*deltaTimeSwaption) + "\t")
        
        for j in range(MaturitySwaptionMax + 1):
            #forwardSwapRate = forwardSwapRateCIR2PP(0, double(j), double((i + 1)) *deltaTimeSwaption, deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, x_0CIR2PP, y_0CIR2PP)
            forwardSwapRate = forwardSwapRateMatrix[j][i]
            ofForwardSwapRateCIR2PP.write(str(forwardSwapRate) + "\t")
        ofForwardSwapRateCIR2PP.write("\n")
        
#le J va servir a incrementer les tables d'outputs
J = 1

#Initialisation des courbes de taux initiale dans les bases d'outputs
#Modele G2++
if modelG2PP == True:
    
    for k in range(nombreMaturity):
        maturiteDate = initialMaturityVecteur[k]
        xFacteur = x_0G2PP
        yFacteur = y_0G2PP
        zeroCouponYieldOutputG2PP[0][k][0] = zeroCouponYieldG2PP(0, 0 + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)
        zeroCouponYieldOutputG2PP[1][k][0] = zeroCouponYieldG2PP(0, 0 + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)
        zeroCouponPriceOutputG2PP[0][k][0] = zeroCouponPriceG2PP(0, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)
        zeroCouponPriceOutputG2PP[1][k][0] = zeroCouponPriceG2PP(0, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)

#Modele CIR2++
if modelCIR2PP == True :
    
    for k in range(nombreMaturity):
        maturiteDate = initialMaturityVecteur[k]
        xFacteur = x_0CIR2PP
        yFacteur = y_0CIR2PP
        zeroCouponYieldOutputCIR2PP[0][k][0] = zeroCouponYieldCIR2PP(0, 0 + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
        zeroCouponYieldOutputCIR2PP[1][k][0] = zeroCouponYieldCIR2PP(0, 0 + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
        zeroCouponPriceOutputCIR2PP[0][k][0] = zeroCouponPriceCIR2PP(0, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
        zeroCouponPriceOutputCIR2PP[1][k][0] = zeroCouponPriceCIR2PP(0, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)

#Sortie de la premiere ligne sur les outputs de Browniens
if sortieBrowniens == True:
    #Generation d'uniformes
    ofBrownien1.write('\t')
    ofBrownien2Uncorrelated.write('\t')
    ofBrownien2Correlated.write('\t')
    for j in range(nombreTimeStep):
        timeIndex = timeIndex + timeStep
        ofBrownien1.write(str(timeIndex)+'\t')
        ofBrownien2Uncorrelated.write(str(timeIndex)+'\t')
        ofBrownien2Correlated.write(str(timeIndex)+'\t')
    ofBrownien1.write('\n')
    ofBrownien2Uncorrelated.write('\n')
    ofBrownien2Correlated.write('\n')

#Sortie de la premiere ligne sur les outputs de deflateurs
if sortieScenarioDeflateur == True:
    timeIndex = 1 / splitAnnuelOutput
    if (modelG2PP == True):
        ofSortieDeflateurG2PP.write("\t" + str(timeIndex) + "\t")
    if (modelCIR2PP == True):
        ofSortieDeflateurCIR2PP.write("\t" + str(timeIndex) + "\t")
    for j in range(nombreTimeStepOutput):
        timeIndex = timeIndex + 1 / splitAnnuelOutput
        if (modelG2PP == True):
            ofSortieDeflateurG2PP.write(str(timeIndex) + "\t")
        if (modelCIR2PP == True):
            ofSortieDeflateurCIR2PP.write(str(timeIndex) + "\t")
    if (modelG2PP == True):
        ofSortieDeflateurG2PP.write("\n")
    if (modelCIR2PP == True):
        ofSortieDeflateurCIR2PP.write("\n")

#Sortie de la premiere ligne sur les outputs de taux ZC
if (sortieScenarioTaux == True):
    timeIndex = 0
    ofSortieTauxCIR2PP.write("Nelson-Siegel-Svensson Model Parameters;\n")
    ofSortieTauxCIR2PP.write("\tlambda_1:\t" + str(lambda_1) + "\n")
    ofSortieTauxCIR2PP.write("\tlambda_2:\t" + str(lambda_2) + "\n")
    ofSortieTauxCIR2PP.write("\tbeta_0:\t" + str(beta_0) + "\n")
    ofSortieTauxCIR2PP.write("\tbeta_1:\t" + str(beta_1) + "\n")
    ofSortieTauxCIR2PP.write("\tbeta_2:\t" + str(beta_2) + "\n")
    ofSortieTauxCIR2PP.write("\tbeta_3:\t" + str(beta_3) + "\n")
    ofSortieTauxCIR2PP.write("CIR2++ Model Parameters:\n")
    ofSortieTauxCIR2PP.write("\tx_0:\t" + str(x_0CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("\tkappa_1:\t" + str(kappa_1CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("\ttheta_1:\t" + str(theta_1CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("\tsigma_1:\t" + str(sigma_1CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("\ty_0:\t" + str(y_0CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("\tkappa_2:\t" + str(kappa_2CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("\ttheta_2:\t" + str(theta_2CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("\tsigma_2:\t" + str(sigma_2CIR2PP) + "\n")
    ofSortieTauxCIR2PP.write("T=\t")
    if (modelG2PP == True):
        ofSortieTauxG2PP.write("\t" + str(timeIndex) + "\t")
    if (modelCIR2PP == True):
        ofSortieTauxCIR2PP.write("\t" + str(timeIndex) + "\t")
    for j in range(nombreTimeStepOutput + 1):
        timeIndex = timeIndex + 1 / splitAnnuelOutput
        if (modelG2PP == True):
            ofSortieTauxG2PP.write(str(timeIndex) + "\t")
        if (modelCIR2PP == True):
            ofSortieTauxCIR2PP.write(str(timeIndex) + "\t")
    if (modelG2PP == True):
        ofSortieTauxG2PP.write("\n")
    if (modelCIR2PP == True):
        ofSortieTauxCIR2PP.write("\n")

#Sortie de la première ligne sur les outputs de prix ZC
if (sortieScenarioPrix ==True):
    timeIndex = 0.0
    if (modelG2PP == True):
        ofSortieDiscountedPrixZeroCouponsG2PP.write("\t" + "\t" + str(timeIndex) + "\t")
    if (modelCIR2PP == True):
        ofSortieDiscountedPrixZeroCouponsCIR2PP.write("\t" + "\t" + str(timeIndex) + "\t")
    for j in range(nombreTimeStepOutput):
        timeIndex = timeIndex + 1 / splitAnnuelOutput
        if (modelG2PP == True):
            ofSortieDiscountedPrixZeroCouponsG2PP.write(str(timeIndex) + "\t")
        if (modelCIR2PP == True):
            ofSortieDiscountedPrixZeroCouponsCIR2PP.write(str(timeIndex) + "\t")
    if (modelG2PP == True):
        ofSortieDiscountedPrixZeroCouponsG2PP.write("\n")
    if (modelCIR2PP == True):
        ofSortieDiscountedPrixZeroCouponsCIR2PP.write("\n")
            
#Sortie de la première ligne sur les outputs de tests martingales (sur les prix actualisés)
if (sortieTestMartingaleMaturiteFixe == True):
    timeIndex = 0.0
    if (modelG2PP == True):
        ofMartingaleTestDiscZCPriceG2PP.write("\t" + "\t" + str(timeIndex) + "\t")
    if (modelCIR2PP == True):
        ofMartingaleTestDiscZCPriceCIR2PP.write("\t" + "\t" + str(timeIndex) + "\t")
    for j in range(nombreTimeStepOutput):
        timeIndex = timeIndex + 1 / splitAnnuelOutput
        if (modelG2PP == True):
            ofMartingaleTestDiscZCPriceG2PP.write(str(timeIndex) + "\t")
        if (modelCIR2PP == True):
            ofMartingaleTestDiscZCPriceCIR2PP.write(str(timeIndex) + "\t")
    if (modelG2PP == True):
        ofMartingaleTestDiscZCPriceG2PP .write("\n")
    if (modelCIR2PP == True):
        ofMartingaleTestDiscZCPriceCIR2PP.write("\n")

#*****************************Debut de la boucle de simulation*****************************

for i in range(nombreSimu // 2): 
    #On s'arrête à nbSimu/2 car on va tirer les antithétiques
    #Attention au décalage, i=0 correspond à la simulation n°1
    #Attention au décalage, j=0 correspond au pas de temps 1 jour
    #Ainsi, il convient d'initialiser les valeurs x_1 et y_1 à partir de x_0G2PP et y_0G2PP qui ne figurent pas dans le tableau
    J = 1 #Je réinitialise le J pour la nouvelle simu
    
    #******************************** Simulations des Browniens sur le premier pas de temps ****************************************
    #Simulation des 3 browniens sur le pas de temps 0
    normaleBrownien1[0][0] = generateur_normale(random.random())*math.sqrt(timeStep)
    normaleBrownien2Uncorrelated[0][0] = generateur_normale(random.random())*math.sqrt(timeStep)
    if modelG2PP == True:
        normaleBrownien2Correlated[0][0] = rhoG2PP*normaleBrownien1[0][0] + math.sqrt(1 - rhoG2PP *rhoG2PP)*normaleBrownien2Uncorrelated[0][0]
    #Simulation des 3 browniens sur la trajectoire antithétique et sur le pas de temps 0
    normaleBrownien1[1][0] = -normaleBrownien1[0][0]
    normaleBrownien2Uncorrelated[1][0] = -normaleBrownien2Uncorrelated[0][0]
    if modelG2PP == True:
        normaleBrownien2Correlated[1][0] = -normaleBrownien2Correlated[0][0]
    
    #******************************** Simulations des facteurs sur le premier pas de temps ****************************************/
    #Simulation des facteurs sur le pas de temps 0, trajectoire centrale
    if (modelG2PP == True):
        
        xFactorG2PP[0][0] = x_0G2PP - coeff_aG2PP*x_0G2PP*timeStep + sigmaG2PP*normaleBrownien1[0][0]
        yFactorG2PP[0][0] = y_0G2PP - coeff_bG2PP*y_0G2PP*timeStep + eta*normaleBrownien2Correlated[0][0]
        rFactorG2PP[0][0] = xFactorG2PP[0][0] + yFactorG2PP[0][0] + functionPhiG2PP(timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, coeff_bG2PP, sigmaG2PP, eta, rhoG2PP)
        deflatorG2PP[0][0] = math.exp(-r_0G2PP*timeStep)
        
        #Simulation des facteurs sur le pas de temps 0, trajectoire antithétique
        xFactorG2PP[1][0] = x_0G2PP - coeff_aG2PP*x_0G2PP*timeStep + sigmaG2PP*normaleBrownien1[1][0]
        yFactorG2PP[1][0] = y_0G2PP - coeff_bG2PP*y_0G2PP*timeStep + eta*normaleBrownien2Correlated[1][0]
        rFactorG2PP[1][0] = xFactorG2PP[1][0] + yFactorG2PP[1][0] + functionPhiG2PP(timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, coeff_bG2PP, sigmaG2PP, eta, rhoG2PP)
        deflatorG2PP[1][0] = math.exp(-r_0G2PP*timeStep)
    
    if (modelCIR2PP == True):
        
        #Schéma Simple
        #xFactorCIR2PP[0][0] = abs(x_0CIR2PP + kappa_1CIR2PP*(theta_1CIR2PP - x_0CIR2PP)*timeStep + sigma_1CIR2PP*math.sqrt(x_0CIR2PP)*normaleBrownien1[0][0])
        #yFactorCIR2PP[0][0] = abs(y_0CIR2PP + kappa_2CIR2PP*(theta_2CIR2PP - y_0CIR2PP)*timeStep + sigma_2CIR2PP*math.sqrt(y_0CIR2PP)*normaleBrownien2Uncorrelated[0][0])
        #Schéma E(0)
        xFactorCIR2PP[0][0] = pow(((1 - kappa_1CIR2PP*timeStep / 2)*math.sqrt(x_0CIR2PP) + sigma_1CIR2PP*normaleBrownien1[0][0] / (2 * (1 - kappa_1CIR2PP*timeStep / 2))), 2) + (kappa_1CIR2PP*theta_1CIR2PP - sigma_1CIR2PP*sigma_1CIR2PP / 4)*timeStep
        yFactorCIR2PP[0][0] = pow(((1 - kappa_2CIR2PP*timeStep / 2)*math.sqrt(y_0CIR2PP) + sigma_2CIR2PP*normaleBrownien2Uncorrelated[0][0] / (2 * (1 - kappa_2CIR2PP*timeStep / 2))), 2) + (kappa_2CIR2PP*theta_2CIR2PP - sigma_2CIR2PP*sigma_2CIR2PP / 4)*timeStep
        rFactorCIR2PP[0][0] = xFactorCIR2PP[0][0] + yFactorCIR2PP[0][0] + functionPhiCIR2PP(timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, x_0CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, y_0CIR2PP);
        deflatorCIR2PP[0][0] = math.exp(-r_0CIR2PP*timeStep);

        #Simulation des facteurs sur le pas de temps 0, trajectoire antithétique
        #Schéma Simple
        #xFactorCIR2PP[1][0] = abs(x_0CIR2PP + kappa_1CIR2PP*(theta_1CIR2PP - x_0CIR2PP)*timeStep + sigma_1CIR2PP*math.sqrt(x_0CIR2PP)*normaleBrownien1[1][0])
        #yFactorCIR2PP[1][0] = abs(y_0CIR2PP + kappa_2CIR2PP*(theta_2CIR2PP - y_0CIR2PP)*timeStep + sigma_2CIR2PP*math.sqrt(y_0CIR2PP)*normaleBrownien2Uncorrelated[1][0])
        #Schéma E(0)
        xFactorCIR2PP[1][0] = pow(((1 - kappa_1CIR2PP*timeStep / 2)*math.sqrt(x_0CIR2PP) + sigma_1CIR2PP*normaleBrownien1[1][0] / (2 * (1 - kappa_1CIR2PP*timeStep / 2))), 2) + (kappa_1CIR2PP*theta_1CIR2PP - sigma_1CIR2PP*sigma_1CIR2PP / 4)*timeStep
        yFactorCIR2PP[1][0] = pow(((1 - kappa_2CIR2PP*timeStep / 2)*math.sqrt(y_0CIR2PP) + sigma_2CIR2PP*normaleBrownien2Uncorrelated[1][0] / (2 * (1 - kappa_2CIR2PP*timeStep / 2))), 2) + (kappa_2CIR2PP*theta_2CIR2PP - sigma_2CIR2PP*sigma_2CIR2PP / 4)*timeStep
        rFactorCIR2PP[1][0] = xFactorCIR2PP[1][0] + yFactorCIR2PP[1][0] + functionPhiCIR2PP(timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, x_0CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, y_0CIR2PP)
        deflatorCIR2PP[1][0] = math.exp(-r_0CIR2PP*timeStep)
        
    timeIndex = 0.0 #Variable servant en cas de sortie des scénarios de taux   
    
    for j in range(1,nombreTimeStep): #Je fais la boucle sur chaque pas de temps
        #******************************** Simulations des Browniens ****************************************
        #Simulation des 3 browniens sur une trajectoire
        normaleBrownien1[0][j] = generateur_normale(random.random())*math.sqrt(timeStep)
        normaleBrownien2Uncorrelated[0][j] = generateur_normale(random.random())*math.sqrt(timeStep)
        if modelG2PP == True:
            normaleBrownien2Correlated[0][j] = rhoG2PP*normaleBrownien1[0][j] + math.sqrt(1 - rhoG2PP *rhoG2PP)*normaleBrownien2Uncorrelated[0][j]
        #Simulation des 3 browniens sur la trajectoire antithétique
        normaleBrownien1[1][j] = -normaleBrownien1[0][j]
        normaleBrownien2Uncorrelated[1][j] = -normaleBrownien2Uncorrelated[0][j]
        if modelG2PP == True:
            normaleBrownien2Correlated[1][j] = -normaleBrownien2Correlated[0][j]
        
        #******************************** Simulations des Facteurs ****************************************
        if (modelG2PP == True):
            
            #Simulations des facteurs du G2++, trajectoire centrale
            xFactorG2PP[0][j] = xFactorG2PP[0][j - 1] - coeff_aG2PP*xFactorG2PP[0][j - 1] * timeStep + sigmaG2PP*normaleBrownien1[0][j]
            yFactorG2PP[0][j] = yFactorG2PP[0][j - 1] - coeff_bG2PP*yFactorG2PP[0][j - 1] * timeStep + eta*normaleBrownien2Correlated[0][j]
            rFactorG2PP[0][j] = xFactorG2PP[0][j] + yFactorG2PP[0][j] + functionPhiG2PP((j + 1) * timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, coeff_bG2PP, sigmaG2PP, eta, rhoG2PP)
            deflatorG2PP[0][j] = math.exp(-rFactorG2PP[0][j - 1] * timeStep)*deflatorG2PP[0][j - 1]
            
            #Simulations des facteurs du G2++, trajectoire antithétique
            xFactorG2PP[1][j] = xFactorG2PP[1][j - 1] - coeff_aG2PP*xFactorG2PP[1][j - 1] * timeStep + sigmaG2PP*normaleBrownien1[1][j]
            yFactorG2PP[1][j] = yFactorG2PP[1][j - 1] - coeff_bG2PP*yFactorG2PP[1][j - 1] * timeStep + eta*normaleBrownien2Correlated[1][j]
            rFactorG2PP[1][j] = xFactorG2PP[1][j] + yFactorG2PP[1][j] + functionPhiG2PP((j + 1) * timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, coeff_bG2PP, sigmaG2PP, eta, rhoG2PP)
            deflatorG2PP[1][j] = math.exp(-rFactorG2PP[1][j - 1] * timeStep)*deflatorG2PP[1][j - 1]
        
        if (modelCIR2PP == True):
            
            #Schéma Simple
            #xFactorCIR2PP[0][j] = abs(xFactorCIR2PP[0][j-1] + kappa_1CIR2PP*(theta_1CIR2PP - xFactorCIR2PP[0][j-1])*timeStep + sigma_1CIR2PP*math.sqrt(xFactorCIR2PP[0][j-1])*normaleBrownien1[0][j])
            #yFactorCIR2PP[0][j] = fabs(yFactorCIR2PP[0][j-1] + kappa_2CIR2PP*(theta_2CIR2PP - yFactorCIR2PP[0][j-1])*timeStep + sigma_2CIR2PP*math.sqrt(yFactorCIR2PP[0][j-1])*normaleBrownien2Uncorrelated[0][j])
            #Schéma E(0)
            xFactorCIR2PP[0][j] = pow(((1 - kappa_1CIR2PP*timeStep / 2)*math.sqrt(xFactorCIR2PP[0][j-1]) + sigma_1CIR2PP*normaleBrownien1[0][j] / (2 * (1 - kappa_1CIR2PP*timeStep / 2))), 2) + (kappa_1CIR2PP*theta_1CIR2PP - sigma_1CIR2PP*sigma_1CIR2PP / 4)*timeStep
            yFactorCIR2PP[0][j] = pow(((1 - kappa_2CIR2PP*timeStep / 2)*math.sqrt(yFactorCIR2PP[0][j-1]) + sigma_2CIR2PP*normaleBrownien2Uncorrelated[0][j] / (2 * (1 - kappa_2CIR2PP*timeStep / 2))), 2) + (kappa_2CIR2PP*theta_2CIR2PP - sigma_2CIR2PP*sigma_2CIR2PP / 4)*timeStep
            rFactorCIR2PP[0][j] = xFactorCIR2PP[0][j] + yFactorCIR2PP[0][j] + functionPhiCIR2PP((j + 1) * timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, x_0CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, y_0CIR2PP)
            deflatorCIR2PP[0][j] = math.exp(-rFactorCIR2PP[0][j - 1] * timeStep)*deflatorCIR2PP[0][j - 1]
            
            #Simulation des facteurs sur le pas de temps 0, trajectoire antithétique
            #Schéma Simple
            #xFactorCIR2PP[1][j] = abs(xFactorCIR2PP[1][j-1] + kappa_1CIR2PP*(theta_1CIR2PP - xFactorCIR2PP[1][j-1])*timeStep + sigma_1CIR2PP*math.sqrt(xFactorCIR2PP[1][j-1])*normaleBrownien1[1][j])
            #yFactorCIR2PP[1][j] = abs(yFactorCIR2PP[1][j-1] + kappa_2CIR2PP*(theta_2CIR2PP - yFactorCIR2PP[1][j-1])*timeStep + sigma_2CIR2PP*math.sqrt(yFactorCIR2PP[1][j-1])*normaleBrownien2Uncorrelated[1][j])
            #Schéma E(0)
            xFactorCIR2PP[1][j] = pow(((1 - kappa_1CIR2PP*timeStep / 2)*math.sqrt(xFactorCIR2PP[1][j-1]) + sigma_1CIR2PP*normaleBrownien1[1][j] / (2 * (1 - kappa_1CIR2PP*timeStep / 2))), 2) + (kappa_1CIR2PP*theta_1CIR2PP - sigma_1CIR2PP*sigma_1CIR2PP / 4)*timeStep
            yFactorCIR2PP[1][j] = pow(((1 - kappa_2CIR2PP*timeStep / 2)*math.sqrt(yFactorCIR2PP[1][j-1]) + sigma_2CIR2PP*normaleBrownien2Uncorrelated[1][j] / (2 * (1 - kappa_2CIR2PP*timeStep / 2))), 2) + (kappa_2CIR2PP*theta_2CIR2PP - sigma_2CIR2PP*sigma_2CIR2PP / 4)*timeStep
            rFactorCIR2PP[1][j] = xFactorCIR2PP[1][j] + yFactorCIR2PP[1][j] + functionPhiCIR2PP((j + 1) * timeStep, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, x_0CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, y_0CIR2PP)
            deflatorCIR2PP[1][j] = math.exp(-rFactorCIR2PP[1][j - 1] * timeStep)*deflatorCIR2PP[1][j - 1]
        
        if (j == int(nombreSimuAnnee / splitAnnuelOutput*(J) - 1)): #Je remplis les facteurs sur les pas d'outputs
            if (modelG2PP == True):
                
                #Sorties des facteurs et déflateurs
                #Sorties des facteurs et déflateurs, trajectoire centrale
                FactorG2PPOutput[0][J - 1] = xFactorG2PP[0][j]
                yFactorG2PPOutput[0][J - 1] = yFactorG2PP[0][j]
                rFactorG2PPOutput[0][J - 1] = rFactorG2PP[0][j]
                deflatorG2PPOutput[0][J - 1] = deflatorG2PP[0][j]
                
                #Sorties des facteurs et déflateurs, trajectoire antithétique
                xFactorG2PPOutput[1][J - 1] = xFactorG2PP[1][j]
                yFactorG2PPOutput[1][J - 1] = yFactorG2PP[1][j]
                rFactorG2PPOutput[1][J - 1] = rFactorG2PP[1][j]
                deflatorG2PPOutput[1][J - 1] = deflatorG2PP[1][j]
            
            if (modelCIR2PP == True):
                
                #Sorties des facteurs et déflateurs
                #Sorties des facteurs et déflateurs, trajectoire centrale
                xFactorCIR2PPOutput[0][J - 1] = xFactorCIR2PP[0][j]
                yFactorCIR2PPOutput[0][J - 1] = yFactorCIR2PP[0][j]
                rFactorCIR2PPOutput[0][J - 1] = rFactorCIR2PP[0][j]
                deflatorCIR2PPOutput[0][J - 1] = deflatorCIR2PP[0][j]
                
                #Sorties des facteurs et déflateurs, trajectoire antithétique
                xFactorCIR2PPOutput[1][J - 1] = xFactorCIR2PP[1][j]
                yFactorCIR2PPOutput[1][J - 1] = yFactorCIR2PP[1][j]
                rFactorCIR2PPOutput[1][J - 1] = rFactorCIR2PP[1][j]
                deflatorCIR2PPOutput[1][J - 1] = deflatorCIR2PP[1][j]
            
            if (sortieTestMartingaleDeflateur == 1): #On incrémente la première ligne pour le test déflateur pour chaque pas de temps
                if (modelG2PP == True):
                    martingaleDiscountG2PP[0][J - 1] = martingaleDiscountG2PP[0][J - 1] + (deflatorG2PPOutput[0][J - 1] + deflatorG2PPOutput[1][J - 1]) / nombreSimu
                if (modelCIR2PP == True):
                    martingaleDiscountCIR2PP[0][J - 1] = martingaleDiscountCIR2PP[0][J - 1] + (deflatorCIR2PPOutput[0][J - 1] + deflatorCIR2PPOutput[1][J - 1]) / nombreSimu
                    
            for k in range(nombreMaturity): #On remplit le taux ZC et le prix ZC pour le central et son antithétique
                tempsDate = J / splitAnnuelOutput
                maturiteDate = initialMaturityVecteur[k]
                if (modelG2PP == True):
                    
                    xFacteur = xFactorG2PPOutput[0][J - 1]
                    yFacteur = yFactorG2PPOutput[0][J - 1]
                    zeroCouponYieldOutputG2PP[0][k][J] = zeroCouponYieldG2PP(tempsDate, tempsDate + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)
                    if (tempsDate < maturiteDate):
                        zeroCouponPriceOutputG2PP[0][k][J] = zeroCouponPriceG2PP(tempsDate, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)
                    
                    xFacteur = xFactorG2PPOutput[1][J - 1]
                    yFacteur = yFactorG2PPOutput[1][J - 1]
                    zeroCouponYieldOutputG2PP[1][k][J] = zeroCouponYieldG2PP(tempsDate, tempsDate + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)
                    if (tempsDate < maturiteDate):
                        zeroCouponPriceOutputG2PP[1][k][J] = zeroCouponPriceG2PP(tempsDate, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, xFacteur, yFacteur)
                        
                if (modelCIR2PP == True):
                    
                    xFacteur = xFactorCIR2PPOutput[0][J - 1]
                    yFacteur = yFactorCIR2PPOutput[0][J - 1]
                    zeroCouponYieldOutputCIR2PP[0][k][J] = zeroCouponYieldCIR2PP(tempsDate, tempsDate + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                    if (tempsDate < maturiteDate):
                        zeroCouponPriceOutputCIR2PP[0][k][J] = zeroCouponPriceCIR2PP(tempsDate, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                    
                    xFacteur = xFactorCIR2PPOutput[1][J - 1]
                    yFacteur = yFactorCIR2PPOutput[1][J - 1]
                    zeroCouponYieldOutputCIR2PP[1][k][J] = zeroCouponYieldCIR2PP(tempsDate, tempsDate + maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                    if (tempsDate < maturiteDate):
                        zeroCouponPriceOutputCIR2PP[1][k][J] = zeroCouponPriceCIR2PP(tempsDate, maturiteDate, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
            
            xFacteur = xFactorCIR2PPOutput[0][J - 1]
            yFacteur = yFactorCIR2PPOutput[0][J - 1]
            #il faut que je compare le taux swap forward à la date J au taux swap de maturity J, pour chacun des tenor et des moneyness
            #Ce taux swap forward vu de zéro est stocké dans la matrice forwardSwapRateMatrix[J][tenor, sachant que le 0 c'est +1an][moneyness]
            #je stocke cela dans pricerSwaptionOutputCIR2PP[J]5tenor][moneyNess]
            #attention penser à diviser le tenor par deltaTimeSwaption
            #if (pricerSwaptionCIR2PP==True):
                #if (J < MaturitySwaptionMax):
                    #discountFactorSwaption = 0
                    #for tenorLocal in range(tenorMax // deltaTimeSwaption):
                        #discountFactorSwaption += deltaTimeSwaption*zeroCouponPriceCIR2PP(J, J + (tenorLocal + 1)*deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                        #forwardSwapRate = forwardSwapRateCIR2PP(J, 0, tenorLocal, deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                        #for moneyLocal in range(moneyNessIndex):
                            #strike = (0.5 + 0.05*moneyLocal)*forwardSwapRateMatrix[J][tenorLocal]
                            #strike = forwardSwapRateMatrix[J][tenorLocal]
                            #pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][moneyLocal] += (payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike)
                            #pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][0] += deflatorCIR2PPOutput[0][J - 1]*(payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike)

            if (pricerSwaptionCIR2PP == True):
                if (J <= MaturitySwaptionMax):
                    discountFactorSwaption = 0
                    for tenorLocal in range(tenorMax//deltaTimeSwaption):
                        discountFactorSwaption += deltaTimeSwaption*zeroCouponPriceCIR2PP(J, J + (tenorLocal + 1)*deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                        forwardSwapRate = (1 - zeroCouponPriceCIR2PP(J, J + (tenorLocal + 1)*deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)) / discountFactorSwaption
                        strike = forwardSwapRateMatrix[J][tenorLocal]
                        pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][0] += deflatorCIR2PPOutput[0][J - 1] * (2 * payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike)
                    #for moneyLocal in range(moneyNessIndex):
                        #strike = (0.5 + 0.05*moneyLocal)*forwardSwapRateMatrix[J][tenorLocal]
                        #pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][moneyLocal] += (payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike)
    
            xFacteur = xFactorCIR2PPOutput[1][J - 1]
            yFacteur = yFactorCIR2PPOutput[1][J - 1]
            #il faut que je compare le taux swap forward à la date J au taux swap de maturity J, pour chacun des tenor et des moneyness
            #Ce taux swap forward vu de zéro est stocké dans la matrice forwardSwapRateMatrix[J][tenor, sachant que le 0 c'est +1an][moneyness]
            #je stocke cela dans pricerSwaptionOutputCIR2PP[J]5tenor][moneyNess]
            #attention penser à diviser le tenor par deltaTimeSwaption
            #if (pricerSwaptionCIR2PP == True):
                #if (J < MaturitySwaptionMax):
                    #discountFactorSwaption = 0
                    #for tenorLocal in range(tenorMax // deltaTimeSwaption):
                        #discountFactorSwaption += deltaTimeSwaption*zeroCouponPriceCIR2PP(J, J + (tenorLocal + 1)*deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                        #forwardSwapRate = forwardSwapRateCIR2PP(J, 0, tenorLocal, deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                        #for moneyLocal in range(moneyNessIndex):
                            #strike = (0.5 + 0.05*moneyLocal)*forwardSwapRateMatrix[J][tenorLocal]
                            #strike = forwardSwapRateMatrix[J][tenorLocal]
                            #pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][moneyLocal] += (payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike)
                            #pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][0] += deflatorCIR2PPOutput[1][J - 1] * (payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike);


            if (pricerSwaptionCIR2PP == True):
                if (J <= MaturitySwaptionMax):
                    discountFactorSwaption = 0
                    for tenorLocal in range(tenorMax // deltaTimeSwaption):
                        discountFactorSwaption += deltaTimeSwaption*zeroCouponPriceCIR2PP(J, J + (tenorLocal + 1)*deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)
                        forwardSwapRate = (1 - zeroCouponPriceCIR2PP(J, J + (tenorLocal + 1)*deltaTimeSwaption, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, xFacteur, yFacteur)) / discountFactorSwaption
                        strike = forwardSwapRateMatrix[J][tenorLocal]
                        pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][0] += deflatorCIR2PPOutput[1][J - 1] * (2 * payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike)
                    
                    #for moneyLocal in range(moneyNessIndex):
                        #strike = (0.5 + 0.05*moneyLocal)*forwardSwapRateMatrix[J][tenorLocal]
                        #pricerSwaptionOutputCIR2PP[J - 1][tenorLocal][moneyLocal] += (payerIndex - 1) * 1 / nombreSimu*discountFactorSwaption*payoffMaxZero(forwardSwapRate - strike)                  
            
            J+=1
    
    for k in range(nombreMaturity): #Je calcule la moyenne empirique sur chacun des deux scénarios pour chaque maturité et pas de temps
        for j in range(int(min(initialMaturityVecteur[k] * splitAnnuelOutput, nombreTimeStepOutput + 1))):
            if (j == 0):
                    
                if (modelG2PP == True):
                    martingaleTestTempG2PP[k][j] += zeroCouponPriceOutputG2PP[0][k][j] / nombreSimu
                    martingaleTestTempG2PP[k][j] += zeroCouponPriceOutputG2PP[1][k][j] / nombreSimu
                        
                if (modelCIR2PP == True):
                    martingaleTestTempCIR2PP[k][j] += zeroCouponPriceOutputCIR2PP[0][k][j] / nombreSimu
                    martingaleTestTempCIR2PP[k][j] += zeroCouponPriceOutputCIR2PP[1][k][j] / nombreSimu
                        
            else:
                    
                if (modelG2PP == True):
                        
                    martingaleTestTempG2PP[k][j] += deflatorG2PPOutput[0][j - 1] * zeroCouponPriceOutputG2PP[0][k][j] / nombreSimu
                    martingaleTestTempG2PP[k][j] += deflatorG2PPOutput[1][j - 1] * zeroCouponPriceOutputG2PP[1][k][j] / nombreSimu

                if (modelCIR2PP == True):
                    martingaleTestTempCIR2PP[k][j] += deflatorCIR2PPOutput[0][j - 1] * zeroCouponPriceOutputCIR2PP[0][k][j] / nombreSimu
                    martingaleTestTempCIR2PP[k][j] += deflatorCIR2PPOutput[1][j - 1] * zeroCouponPriceOutputCIR2PP[1][k][j] / nombreSimu
                        

    #SORTIE des browniens au format .txt
    if (sortieBrowniens == True):
        ofBrownien1.write("Simu"+  str(2*i + 1) + "\t")
        ofBrownien2Uncorrelated.write("Simu" + str(2 * i + 1) + "\t")
        if modelG2PP == True:
            ofBrownien2Correlated.write("Simu" + str(2 * i + 1) + "\t")
        for j in range(nombreTimeStep):
            ofBrownien1.write(str(normaleBrownien1[0][j]) + "\t")
            ofBrownien2Uncorrelated.write(str(normaleBrownien2Uncorrelated[0][j]) + "\t")
            if modelG2PP == True:
                ofBrownien2Correlated.write(str(normaleBrownien2Correlated[0][j]) + "\t")
        
        ofBrownien1.write("\nSimu" + str(2 * (i + 1)) + "\t")
        ofBrownien2Uncorrelated.write("\nSimu" + str(2 * (i + 1)) + "\t")
        if modelG2PP == True:
            ofBrownien2Correlated.write("\nSimu" + str(2 * (i + 1)) + "\t")
        for j in range(nombreTimeStep):
            ofBrownien1.write(str(normaleBrownien1[1][j]) + "\t")
            ofBrownien2Uncorrelated.write(str(normaleBrownien2Uncorrelated[1][j]) + "\t")
            if modelG2PP == True:
                ofBrownien2Correlated.write(str(normaleBrownien2Correlated[1][j]) + "\t")
            
        ofBrownien1.write("\n")
        ofBrownien2Uncorrelated.write("\n")
        if modelG2PP == True:
            ofBrownien2Correlated.write("\n")
        
    #SORTIE / Scénario de déflateurs
    if (sortieScenarioDeflateur == True):
        if (modelG2PP == True):
            ofSortieDeflateurG2PP.write("Simu" + str(2 * i + 1) + "\t")
        if (modelCIR2PP == True):
            ofSortieDeflateurCIR2PP.write("Simu" +str(2 * i + 1) + "\t")
        for j in range(nombreTimeStepOutput):
            
            if (modelG2PP == True):
                ofSortieDeflateurG2PP.write(str(deflatorG2PPOutput[0][j]) + "\t")
            if (modelCIR2PP == True):
                ofSortieDeflateurCIR2PP.write(str(deflatorCIR2PPOutput[0][j]) + "\t")
        
        if (modelG2PP == True):
            ofSortieDeflateurG2PP.write("\n")
        if (modelG2PP == True):
            ofSortieDeflateurG2PP.write("Simu" + str(2 * (i + 1)) + "\t")
        if (modelCIR2PP == True):
            ofSortieDeflateurCIR2PP.write("\n")
        if (modelCIR2PP == True):
            ofSortieDeflateurCIR2PP.write("Simu" + str(2 * (i + 1)) + "\t")
        for j in range(nombreTimeStepOutput):
            
            if (modelG2PP == True):
                ofSortieDeflateurG2PP.write(str(deflatorG2PPOutput[1][j]) + "\t")
            if (modelCIR2PP == True):
                ofSortieDeflateurCIR2PP.write(str(deflatorCIR2PPOutput[1][j]) + "\t")
        
        if (modelG2PP == True): 
            ofSortieDeflateurG2PP.write("\n")
        if (modelCIR2PP == True):
            ofSortieDeflateurCIR2PP.write("\n")
    
    
    #SORTIE / Scénario de taux Zero Coupons
    if (sortieScenarioTaux == True):
        if (modelG2PP == True):
            
            for k in range(nombreMaturity):
                ofSortieTauxG2PP.write("Simu" + str(2 * i + 1) + "\tL(t,t+" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(nombreTimeStepOutput + 1):
                    ofSortieTauxG2PP.write(str(zeroCouponYieldOutputG2PP[0][k][j]) + "\t")
                ofSortieTauxG2PP.write("\n")
            
            for k in range(nombreMaturity):
                ofSortieTauxG2PP.write("Simu" + str(2 * (i + 1)) + "\tL(t,t+" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(nombreTimeStepOutput + 1):
                    ofSortieTauxG2PP.write(str(zeroCouponYieldOutputG2PP[1][k][j]) + "\t")
                ofSortieTauxG2PP.write("\n")
                
        if (modelCIR2PP == True):
            
            for k in range(nombreMaturity):
                ofSortieTauxCIR2PP.write("Simu" + str(2 * i + 1) + "\tL(t,t+" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(nombreTimeStepOutput + 1):
                    ofSortieTauxCIR2PP.write(str(zeroCouponYieldOutputCIR2PP[0][k][j]) + "\t")
                ofSortieTauxCIR2PP.write("\n")
                
            for k in range(nombreMaturity):
                ofSortieTauxCIR2PP.write("Simu" + str(2 * (i + 1)) + "\tL(t,t+" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(nombreTimeStepOutput + 1):
                    ofSortieTauxCIR2PP.write(str(zeroCouponYieldOutputCIR2PP[1][k][j]) + "\t")
                ofSortieTauxCIR2PP.write("\n")


    #SORTIE / Scénario de Prix Zero Coupons
    if (sortieScenarioPrix == True):
        if (modelG2PP == True):
            
            for k in range(nombreMaturity): #Sortie sur le central
                ofSortieDiscountedPrixZeroCouponsG2PP.write("Simu" + str(2 * i + 1) + "\ttilde P(t,T=" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(int(min(initialMaturityVecteur[k] * splitAnnuelOutput, nombreTimeStepOutput + 1))):
                    if (j == 0):
                        ofSortieDiscountedPrixZeroCouponsG2PP.write(str(zeroCouponPriceOutputG2PP[0][k][j]) + "\t")
                    else:
                        ofSortieDiscountedPrixZeroCouponsG2PP.rite(str(deflatorG2PPOutput[0][j - 1] * zeroCouponPriceOutputG2PP[0][k][j]) + "\t")
                ofSortieDiscountedPrixZeroCouponsG2PP.write("\n")
                
            for k in range(nombreMaturity): #Sortie sur l'antithétique
                ofSortieDiscountedPrixZeroCouponsG2PP.write("Simu" + str(2 * (i + 1)) + "\ttilde P(t,T=" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(int(min(initialMaturityVecteur[k] * splitAnnuelOutput, nombreTimeStepOutput + 1))):
                    if (j == 0):
                        ofSortieDiscountedPrixZeroCouponsG2PP.write(str(zeroCouponPriceOutputG2PP[1][k][j]) + "\t")
                    else:
                        ofSortieDiscountedPrixZeroCouponsG2PP.write(str(deflatorG2PPOutput[1][j - 1] * zeroCouponPriceOutputG2PP[1][k][j]) + "\t")
                ofSortieDiscountedPrixZeroCouponsG2PP.write("\n")

        if (modelCIR2PP == True):
            for k in range(nombreMaturity): #Sortie sur le central
                ofSortieDiscountedPrixZeroCouponsCIR2PP.write("Simu" + str(2 * i + 1) + "\ttilde P(t,T=" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(int(min(initialMaturityVecteur[k] * splitAnnuelOutput, nombreTimeStepOutput + 1))):
                    if (j == 0):
                        ofSortieDiscountedPrixZeroCouponsCIR2PP.write(str(zeroCouponPriceOutputCIR2PP[0][k][j]) + "\t")
                    else:
                        ofSortieDiscountedPrixZeroCouponsCIR2PP.write(str(deflatorCIR2PPOutput[0][j - 1] * zeroCouponPriceOutputCIR2PP[0][k][j]) + "\t")
                ofSortieDiscountedPrixZeroCouponsCIR2PP.write("\n")

            for k in range(nombreMaturity): #Sortie sur l'antithétique
                ofSortieDiscountedPrixZeroCouponsCIR2PP.write("Simu" + str(2 * (i + 1)) + "\ttilde P(t,T=" + str(initialMaturityVecteur[k]) + ")\t")
                for j in range(int(min(initialMaturityVecteur[k] * splitAnnuelOutput, nombreTimeStepOutput + 1))):
                    if (j == 0):
                        ofSortieDiscountedPrixZeroCouponsCIR2PP.write(str(zeroCouponPriceOutputCIR2PP[1][k][j]) + "\t")
                    else:
                        ofSortieDiscountedPrixZeroCouponsCIR2PP.write(str(deflatorCIR2PPOutput[1][j - 1] * zeroCouponPriceOutputCIR2PP[1][k][j]) + "\t")
                ofSortieDiscountedPrixZeroCouponsCIR2PP.write("\n")


#******************************** FIN de la boucle de simulation ****************************************               
    

if (pricerSwaptionCIR2PP == True):
    ofPricerSwaptionCIR2PP.write("T\t")
    for j in range(MaturitySwaptionMax):
        ofPricerSwaptionCIR2PP.write(str(j) + "\t")
    ofPricerSwaptionCIR2PP.write("\n")
    for tenorLocal in range(tenorMax // deltaTimeSwaption):
        ofPricerSwaptionCIR2PP.write("T+" + str(tenorLocal+1) + "\t")
        for j in  range(MaturitySwaptionMax):
            ofPricerSwaptionCIR2PP.write(str(pricerSwaptionOutputCIR2PP[j][tenorLocal][0]) + "\t")
        ofPricerSwaptionCIR2PP.write("\n")

#SORTIE / Test martingale déflateur
#/*
#Je crée un tableau martingaleDiscountG2PP dont la première ligne est la moyenne pour chaque pas sur toutes les simus,
#la deuxième ligne est le prix initiale d'un Zero Coupon de maturité le pas de temps considéré
#la deuxième ligne est le backtest qui compare cette moyenne au prix ZC théorique
#*/

if (sortieTestMartingaleDeflateur == True):
    if (modelG2PP == True):
        
        for j in range(nombreTimeStepOutput): #On renseigne les tests entre la moyenne des déflateurs (pour i=0) à chaque pas de temps
            martingaleDiscountG2PP[1][j] = marketInitialZeroCouponPrice(((j + 1)) / splitAnnuelOutput, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3)
            martingaleDiscountG2PP[2][j] = (martingaleDiscountG2PP[1][j] - martingaleDiscountG2PP[0][j]) / martingaleDiscountG2PP[1][j]
            martingaleDiscountG2PP[3][j] = zeroCouponPriceG2PP(0, ((j + 1)) / splitAnnuelOutput, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, sigmaG2PP, coeff_bG2PP, eta, rhoG2PP, x_0G2PP, y_0G2PP)

        ofmartingaleDiscountG2PP = open("ofmartingaleDiscountG2PP.txt", 'w')
        timeIndex = 0.0
        ofmartingaleDiscountG2PP.write("Time steps\t")
        for j in range(nombreTimeStepOutput):
            timeIndex = timeIndex + 1 / splitAnnuelOutput
            ofmartingaleDiscountG2PP.write(str(timeIndex) + "\t")
        ofmartingaleDiscountG2PP.write("\n")
        for i in range(4):
            if (i == 0):
                ofmartingaleDiscountG2PP.write("Simulations empirical mean\t")
            if (i == 1):
                ofmartingaleDiscountG2PP.write("NSS ZC Prices\t")
            if (i == 2): 
                ofmartingaleDiscountG2PP.write("2 first lines error\t")
            if (i == 3):
                ofmartingaleDiscountG2PP.write("Model ZC Prices\t")
            for j in range(nombreTimeStepOutput):
                ofmartingaleDiscountG2PP.write(str(martingaleDiscountG2PP[i][j]) + "\t")
            ofmartingaleDiscountG2PP.write("\n")
        ofmartingaleDiscountG2PP.close()
    
    if (modelCIR2PP == True):
        for j in range(nombreTimeStepOutput): #On renseigne les tests entre la moyenne des déflateurs (pour i=0) à chaque pas de temps
            martingaleDiscountCIR2PP[1][j] = marketInitialZeroCouponPrice(((j + 1)) / splitAnnuelOutput, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3)
            martingaleDiscountCIR2PP[2][j] = (martingaleDiscountCIR2PP[1][j] - martingaleDiscountCIR2PP[0][j]) / martingaleDiscountCIR2PP[1][j]
            martingaleDiscountCIR2PP[3][j] = zeroCouponPriceCIR2PP(0, ((j + 1)) / splitAnnuelOutput, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, kappa_1CIR2PP, theta_1CIR2PP, sigma_1CIR2PP, kappa_2CIR2PP, theta_2CIR2PP, sigma_2CIR2PP, x_0CIR2PP, y_0CIR2PP, x_0CIR2PP, y_0CIR2PP)

        ofmartingaleDiscountCIR2PP = open("ofmartingaleDiscountCIR2PP.txt", 'w')
        timeIndex = 0.0
        ofmartingaleDiscountCIR2PP.write("Time steps\t")
        for j in range(nombreTimeStepOutput):
            timeIndex = timeIndex + 1 / splitAnnuelOutput
            ofmartingaleDiscountCIR2PP.write(str(timeIndex) + "\t")
        ofmartingaleDiscountCIR2PP.write("\n")
        for i in range(4):
            if (i == 0):
                ofmartingaleDiscountCIR2PP.write("Simulations empirical mean\t")
            if (i == 1):
                ofmartingaleDiscountCIR2PP.write("NSS ZC Prices\t")
            if (i == 2):
                ofmartingaleDiscountCIR2PP.write("2 first lines error\t")
            if (i == 3):
                ofmartingaleDiscountCIR2PP.write("Model ZC Prices\t")
            for j in range(nombreTimeStepOutput):
                ofmartingaleDiscountCIR2PP.write(str(martingaleDiscountCIR2PP[i][j]) + "\t")
            ofmartingaleDiscountCIR2PP.write("\n")
        ofmartingaleDiscountCIR2PP.close()
    
    
#SORTIE / Martingale test, maturités fixes 
#IL FAUDRAIT CREER UN TABLEAU
if (sortieTestMartingaleMaturiteFixe == True):
    for k in range(nombreMaturity):
        if (modelG2PP == True):
            ofMartingaleTestDiscZCPriceG2PP.write("\ttilde P(t,T=" + str(initialMaturityVecteur[k]) + ")\t")
        if (modelCIR2PP == True):
            ofMartingaleTestDiscZCPriceCIR2PP.write("\ttilde P(t,T=" + str(initialMaturityVecteur[k]) + ")\t")
        for j in range(int(min(initialMaturityVecteur[k] * splitAnnuelOutput, nombreTimeStepOutput + 1))):
            if (modelG2PP == True):
                ofMartingaleTestDiscZCPriceG2PP.write(str(martingaleTestTempG2PP[k][j]) + "\t")
            if (modelCIR2PP == True):
                ofMartingaleTestDiscZCPriceCIR2PP.write(str(martingaleTestTempCIR2PP[k][j]) + "\t")
        if (modelG2PP == True):
            ofMartingaleTestDiscZCPriceG2PP.write("\n")
        if (modelCIR2PP == True):
            ofMartingaleTestDiscZCPriceCIR2PP.write("\n")

#BACKTEST  du NSS

if (backtestNSS == True):
    initialZeroCouponPrice = np.zeros(120)
    ofzcNSS = open("ofzcNSS.txt",'w')
    for j in range(120):
        ofzcNSS.write(str(((j + 1)) / splitAnnuelOutput) + "\t")
        initialZeroCouponPrice[j] = marketInitialZeroCouponPrice(((j + 1)) / splitAnnuelOutput, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3)
    ofzcNSS.write("\n")
    for j in range(120):
        ofzcNSS.write(str(initialZeroCouponPrice[j]) + "\t")
    ofzcNSS.close()
    
#BACKTEST  de phiG2PP
if (backtestPhiG2PP == True):
    phiG2PP = np.zeros(120)
    ofPhiG2PP = open("ofPhiG2PP.txt",'w')
    for j in range(splitAnnuelOutput*anneeProjection):
        phiG2PP[j] = functionPhiG2PP(((j + 1))/splitAnnuelOutput, lambda_1, lambda_2, beta_0, beta_1, beta_2, beta_3, coeff_aG2PP, coeff_bG2PP, sigmaG2PP, eta, rhoG2PP)
        ofPhiG2PP.write(str(((j + 1))/splitAnnuelOutput) + "\t")
        ofPhiG2PP.write(str(phiG2PP[j]) +  "\n")
    ofPhiG2PP.close()

#Fermeture des fichiers
if sortieBrowniens == True:
    ofBrownien1.close()
    ofBrownien2Uncorrelated.close()
    ofBrownien2Correlated.close()

#Sorties des tables G2++
if modelG2PP == True:
    ofSortieDeflateurG2PP.close()

    ofSortieTauxG2PP.close()

    ofSortieDiscountedPrixZeroCouponsG2PP.close()

    ofMartingaleTestDiscZCPriceG2PP.close()

    ofPricerSwaptionG2PP.close()

#Sorties de tables CIR2++
if modelCIR2PP == True:
    ofSortieDeflateurCIR2PP.close()

    ofSortieTauxCIR2PP.close()

    ofSortieDiscountedPrixZeroCouponsCIR2PP.close()

    ofMartingaleTestDiscZCPriceCIR2PP.close()

    ofForwardSwapRateCIR2PP.close()

    ofPricerSwaptionCIR2PP.close()


#Definition du code de fin pour determiner le temps passe
end = time.perf_counter()
elapsed = end - start
print("Total Time elapsed ", elapsed, " seconds")
