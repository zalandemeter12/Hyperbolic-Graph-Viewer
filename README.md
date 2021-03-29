# Hyperbolic-Graph-Viewer
## Feladat specifikáció 
Készítsen programot, amely egy véletlen gráfot esztétikusan megjelenít és lehetőséget ad a felhasználónak annak tetszőleges részének kinagyítására, mialatt a maradék rész még mindig látszik. A gráf 50 csomópontból áll, telítettsége 5%-os (a lehetséges élek 5% valódi él). Az esztétikus elrendezés érdekében a csomópontok helyét egyrészt heurisztikával, másrészt a hiperbolikus sík szabályainak megfelelő erő-vezérelt gráfrajzoló algoritmussal kell meghatározni a SPACE lenyomásának hatására.

A fókuszálás érdekében a gráfot a hiperbolikus síkon kell elrendezni és a Beltrami-Klein módszerrel a képernyőre vetíteni. A fókuszálás úgy történik, hogy a gráfot a hiperbolikus síkon eltoljuk úgy, hogy az érdekes rész a hiperboloid aljára kerüljön. Az eltolás képi vetülete az egér jobb gombjának lenyomása és lenyomott állapotbeli egérmozgatás pillanatnyi helyének a különbsége.

Az egyes csomópontok a hiperbolikus sík körei, amelyek a csomópontot azonosító textúrával bírnak.

## Futási eredmények
### Raw input:
![enter image description here](https://i.imgur.com/LbEcYDF.png)
### Force directed output:
![enter image description here](https://i.imgur.com/TbimNrk.png)
