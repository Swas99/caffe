#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
FILE=${HERE}/axhm-perf.txt
RUNS="4_6_9 8_8_8 13_13_13 16_8_13"

if [ "" != "$1" ]; then
  FILE=$1
  shift
fi
cat /dev/null > ${FILE}

NRUN=1
NMAX=$(${ECHO} ${RUNS} | wc -w)
for RUN in ${RUNS} ; do
  MVALUE=$(${ECHO} ${RUN} | cut -d_ -f1)
  NVALUE=$(${ECHO} ${RUN} | cut -d_ -f2)
  KVALUE=$(${ECHO} ${RUN} | cut -d_ -f3)
  >&2 ${ECHO} -n "${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})... "
  ERROR=$({ CHECK=1 ${HERE}/axhm.sh ${MVALUE} ${NVALUE} ${KVALUE} $* >> ${FILE}; } 2>&1)
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} "FAILED(${RESULT}) ${ERROR}"
    exit 1
  else
    ${ECHO} "OK ${ERROR}"
  fi
  ${ECHO} >> ${FILE}
  NRUN=$((NRUN+1))
done

