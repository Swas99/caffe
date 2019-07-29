#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
FILE=${HERE}/rstr-perf.txt
RUNS="4_4_4 8_8_8"
RUNT="7_7_7 10_10_10"

if [ "" != "$1" ]; then
  FILE=$1
  shift
fi
cat /dev/null > ${FILE}

NRUN=1
NRUNS=$(${ECHO} ${RUNS} | wc -w)
NRUNT=$(${ECHO} ${RUNT} | wc -w)
NMAX=$((NRUNS*NRUNT))
for RUN1 in ${RUNS} ; do
  for RUN2 in ${RUNT} ; do
  MVALUE=$(${ECHO} ${RUN1} | cut -d_ -f1)
  NVALUE=$(${ECHO} ${RUN1} | cut -d_ -f2)
  KVALUE=$(${ECHO} ${RUN1} | cut -d_ -f3)
  MMVALUE=$(${ECHO} ${RUN2} | cut -d_ -f1)
  NNVALUE=$(${ECHO} ${RUN2} | cut -d_ -f2)
  KKVALUE=$(${ECHO} ${RUN2} | cut -d_ -f3)
  >&2 ${ECHO} -n "${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})... "
  ERROR=$({ CHECK=1 ${HERE}/rstr.sh ${MVALUE} ${NVALUE} ${KVALUE} ${MMVALUE} ${NNVALUE} ${KKVALUE} $* >> ${FILE}; } 2>&1)
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
done

