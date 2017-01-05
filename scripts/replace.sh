for f in *.lua
do
    cat $f |  sed "s|$1|$2|g" > /tmp/x
    echo >> /tmp/x
    cp /tmp/x $f
done

