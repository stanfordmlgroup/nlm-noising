# Adapted from https://raw.githubusercontent.com/facebook/SCRNNs/master/data/makedata-text8.sh
wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f

head text8 -c 90000000 > train
tail text8 -c 10000000 > validtest
head validtest -c 5000000 > valid
tail validtest -c 5000000 > ftest

tr " " '\n' < train | sort | uniq -c | awk '{if ($1>10) print $2;}' | tr "\n" " " > voc

cat voc > trainx
echo $'' >> trainx
cat train >> trainx
awk '{if (NR == 1) {for (a=1;a<=NF;a++) cn[$a]++; b=0;} else for (a=1;a<=NF;a++) {b++; if ((b%1000) == 0) print ""; if (cn[$a]) printf $a " "; else printf "<UNK> ";}}' < trainx > train2
mv train2 text8.train.txt
rm trainx

cat voc > validx
echo $'' >> validx
cat valid >> validx
awk '{if (NR == 1) {for (a=1;a<=NF;a++) cn[$a]++; b=0;} else for (a=1;a<=NF;a++) {b++; if ((b%1000) == 0) print ""; if (cn[$a]) printf $a " "; else printf "<UNK> ";}}' < validx > valid2
mv valid2 text8.valid.txt
rm validx

cat voc > testx
echo $'' >> testx
cat ftest >> testx
awk '{if (NR == 1) {for (a=1;a<=NF;a++) cn[$a]++; b=0;} else for (a=1;a<=NF;a++) {b++; if ((b%1000) == 0) print ""; if (cn[$a]) printf $a " "; else printf "<UNK> ";}}' < testx > test2
mv test2 text8.test.txt
rm testx

#rm voc
#rm train
#rm valid
#rm ftest
#rm text8
