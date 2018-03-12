#!/usr/bin/env bash

mvn clean package
rsync --update -P -h -e ssh ./target/entitylinker-1.0-SNAPSHOT-bin.jar firasd@vilde.cs.lth.se:~/nerd.jar
