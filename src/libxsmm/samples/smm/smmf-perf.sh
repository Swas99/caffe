#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
FILE=${HERE}/smmf-perf.txt
RUNS="23_23_23 4_6_9 13_5_7 24_3_36"

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
  ERROR=$({ CHECK=1 ${HERE}/smm.sh ${MVALUE} ${NVALUE} ${KVALUE} $* >> ${FILE}; } 2>&1)
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

