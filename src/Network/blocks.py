# blocchi.py
# Questo file contiene i "blocchi" modulari riutilizzabili della CNN:
# - ConvBlock: estrae feature dalle immagini
# - ClassifierHead: decide la classe finale

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Blocchetto standard di convoluzione per immagini:
    Conv2d → BatchNorm2d → ReLU → MaxPool2d.

    Serve per estrarre caratteristiche (feature) sempre più complesse.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # nn.Sequential permette di mettere vari layer in sequenza
        self.block = nn.Sequential(  #self.block sarà un “mini-modello” che, 
            #quando chiamato, applica tutti i layer nell’ordine dato.

            # 1) CONVOLUZIONE 2D
            # - in_channels: numero di canali in input (es. 3 per immagini RGB)
            # - out_channels: numero di filtri che vogliamo usare=numero di canali in output
            # - kernel_size=3: filtro 3x3
            # - padding=1: mantiene identica la dimensione spaziale, 
            # aggiunge un bordo di 1 pixel di zeri attorno all’immagine, così che
            #  la dimensione spaziale resta la stessa
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

            # 2) BATCH NORMALIZATION
            # Normalizza ogni canale dell’output della conv.
            # Serve a stabilizzare la distribuzione delle attivazioni.
            nn.BatchNorm2d(out_channels),

            # 3) ReLU: introduce non linearità e mantiene solo ciò che il flltro ritiene utile
            nn.ReLU(),

            # 4) MAX POOLING
            # Riduce la dimensione spaziale (H, W) della metà
            #prende finestre 2x2 e, per ogni finestra, tiene solo il valore massimo.
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        # Applica tutti i layer in blocco
        return self.block(x)



"""class ClassifierHead(nn.Module):
    #Riceve il vettore flatten e produce i logits finali.
    #Scopo: prendere il vettore 1D ottenuto dal flatten del blocco convoluzionale e 
    #trasformarlo in output finale per la classificazione.
    #“Logits” = valori reali prima della softmax
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        # input_dim: dimensione del vettore in input. 
        # Per ogni immagine, l’informazione è compressa in un unico vettore di lunghezza:
        # C(numero canali finali) * H(finale) * W(finale)
        # num_classes: numero di classi di output (4 in questo caso)


        self.classifier = nn.Sequential(

            # Primo layer fully-connected:
            # è fully connected perché così ogni neurone del livello successivo:
            # *riceve in input tutte le feature dell’immagine
            # *può imparare qualsiasi combinazione lineare delle feature precedenti
            # Trasforma il vettore enorme (es. 16384) in 128 feature più compatte.
            # Serve a comprimere il vettore enorme di feature in una rappresentazione 
            # più compatta e utile per la classificazione.
            nn.Linear(input_dim, 128),

            # Attivazione ReLU sul livello nascosto
            nn.ReLU(),

            # Dropout = spegne random il 20% dei neuroni durante il training
            # (serve a ridurre overfitting)
            nn.Dropout(0.2),

            # Layer finale di output: 
            #Trasforma le 128 feature nascoste in 4 valori di output. Come?
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    #la forward applica semplicemente la sequenza di layer."""
