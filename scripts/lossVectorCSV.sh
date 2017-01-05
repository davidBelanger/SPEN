grep 'avg loss' $1 | awk '{print $6}' | tr '\n' ',' | sed 's|,$||' > loss.csv
