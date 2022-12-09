# Thesis
 
 ## Dipartimento di Ingegneria Elettrica e dell'Informazione
 
 ## Autore: Pietro URSO
 
 
 # Author
 
Nessuno è al sicuro sul web, gli attacchi informatici sono sempre più frequenti e le aziende, soprattutto quelle di grandi dimensioni, sono sotto l'occhio dei cyber criminali. 

Inoltre, le tecnologie cloud si sono diffuse in modo esponenziale nel mondo, per cui sono emerse grandi preoccupazioni riguardo alla sicurezza informatica e alla privacy delle aziende e anche delle singole persone che utilizzano i servizi dei provider cloud. Il numero di attacchi informatici lanciati è in costante aumento, questi vengono lanciati su infrastrutture cloud, strutture IT e persino dispositivi IoT; non è solo il numero di attacchi che aumenta con il passare del tempo, ma sono anche le tecniche di attacco che si evolvono, ne nascono di nuove e altre cambiano la loro natura di comportamento nel tempo. 

Infatti, secondo quanto riportato nel Clusit Report 2022, nel 2021 gli attacchi in tutto il mondo sono aumentati del 10% rispetto all'anno precedente, e stanno diventando sempre più gravi. Le nuove modalità di attacco dimostrano che i cybercriminali sono sempre più sofisticati e in grado di fare rete con la criminalità organizzata. La gravità degli attacchi è in netto aumento. Nel 2021, infatti, il 79% degli attacchi rilevati ha avuto un impatto "elevato", rispetto al 50% dello scorso anno. La motivazione principale è la criminalità informatica, che rappresenta l'86% degli attacchi informatici, in aumento rispetto all'81% del 2020. Questi numeri dovrebbero far riflettere: le persone riguardo la vulnerabilità della loro privacy, le aziende riguardo vulnerabilità della loro proprietà intellettuale e del loro patrimonio informativo. 
Nel linguaggio comune i criminali informatici sono chiamati hacker, ma il termine corretto è cracker. Infatti, l'hacker è colui che utilizza le proprie competenze e conoscenze per esplorare e imparare senza causare danni a persone e aziende, mentre il cracker è colui che agisce illegalmente per causare danni o profitti. 
Questo progetto di tesi mira a fornire una soluzione tramite dei modelli di Deep Learning per riconoscere e bloccare eventuali attacchi informatici di diversa natura. Dopo un’attenta analisi sulle soluzioni disponibili presenti ad oggi, i modelli utilizzati, ritenuti i migliori per lo scopo del lavoro di tesi, fanno riferimento a due diversi tipi di approcci:

  •	LSTM Neural Network;
  
  •	Random Forest Classifier. 

Il primo ha consentito di ottenere un modello migliore in termini di accuracy ma molto più dispendioso in termini di capacità computazionale rispetto al secondo. 
Il dataset utilizzato per la costruzione dei due modelli è stato l’UNSW-NB15, pubblicato nel 2015, in Australia, dall’Università del ‘New South Wales’, contenente diverse informazioni, utili alla classificazione dei diversi attacchi nelle varie categorie.

Per la valutazione dei modelli, oltre al valore di accuracy, per entrambe le soluzioni, è stata calcolata la confusion matrix unita ai valori di precision e recall. Tutti questi risultati hanno consentito di individuare il miglior modello, in grado di individuare i diversi tipi di attacchi.

Le soluzioni mostrate all’interno del lavoro di tesi, sono state ottenute mediante l’utilizzo di un PC con le seguenti caratteristiche:

  	CPU: Intel Core i7-8750H - 2.20GHz;   
  
  	RAM: 16 GB;  
  
  	Sistema Operativo: Windows 11; 
  
  	IDE: PyCharm 2021.
