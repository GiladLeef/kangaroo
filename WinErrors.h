// WSA Error string
typedef struct {
  int   errCode;
  char *mesg;
} WSAMAP;

#define NBWSAERRORS 95

WSAMAP WSAERRORS[] = {
  { /*WSA_INVALID_HANDLE*/6,"Specified event object handle is invalid" },
  { /*WSA_NOT_ENOUGH_MEMORY*/8,"Insufficient memory available" },
  { /*WSA_INVALID_PARAMETER*/87,"One or more parameters are invalid" },
  { /*WSA_OPERATION_ABORTED*/995,"Overlapped operation aborted" },
  { /*WSA_IO_INCOMPLETE*/996,"Overlapped I / O event object not in signaled state" },
  { /*WSA_IO_PENDING*/997,"Overlapped operations will complete later" },
  { WSAEINTR,"Interrupted function call" },
  { WSAEBADF,"File handle is not valid" },
  { WSAEACCES,"Permission denied" },
  { WSAEFAULT,"Bad address" },
  { WSAEINVAL,"Invalid argument" },
  { WSAEMFILE,"Too many open files" },
  { WSAEWOULDBLOCK,"Resource temporarily unavailable" },
  { WSAEINPROGRESS,"Operation now in progress" },
  { WSAEALREADY,"Operation already in progress" },
  { WSAENOTSOCK,"Socket operation on nonsocket" },
  { WSAEDESTADDRREQ,"Destination address required" },
  { WSAEMSGSIZE,"Message too long" },
  { WSAEPROTOTYPE,"Protocol wrong type for socket" },
  { WSAENOPROTOOPT,"Bad protocol option" },
  { WSAEPROTONOSUPPORT,"Protocol not supported" },
  { WSAESOCKTNOSUPPORT,"Socket type not supported" },
  { WSAEOPNOTSUPP,"Operation not supported" },
  { WSAEPFNOSUPPORT,"Protocol family not supported" },
  { WSAEAFNOSUPPORT,"Address family not supported by protocol family" },
  { WSAEADDRINUSE,"Address already in use" },
  { WSAEADDRNOTAVAIL,"Cannot assign requested address" },
  { WSAENETDOWN,"Network is down" },
  { WSAENETUNREACH,"Network is unreachable" },
  { WSAENETRESET,"Network dropped connection on reset" },
  { WSAECONNABORTED,"Software caused connection abort" },
  { WSAECONNRESET,"Connection reset by peer" },
  { WSAENOBUFS,"No buffer space available" },
  { WSAEISCONN,"Socket is already connected" },
  { WSAENOTCONN,"Socket is not connected" },
  { WSAESHUTDOWN,"Cannot send after socket shutdown" },
  { WSAETOOMANYREFS,"Too many references" },
  { WSAETIMEDOUT,"Connection timed out" },
  { WSAECONNREFUSED,"Connection refused" },
  { WSAELOOP,"Cannot translate name" },
  { WSAENAMETOOLONG,"Name too long" },
  { WSAEHOSTDOWN,"Host is down" },
  { WSAEHOSTUNREACH,"No route to host" },
  { WSAENOTEMPTY,"Directory not empty" },
  { WSAEPROCLIM,"Too many processes" },
  { WSAEUSERS,"User quota exceeded" },
  { WSAEDQUOT,"Disk quota exceeded" },
  { WSAESTALE,"Stale file handle reference" },
  { WSAEREMOTE,"Item is remote" },
  { WSASYSNOTREADY,"Network subsystem is unavailable" },
  { WSAVERNOTSUPPORTED,"Winsock.dll version out of range" },
  { WSANOTINITIALISED,"Successful WSAStartup not yet performed" },
  { WSAEDISCON,"Graceful shutdown in progress" },
  { WSAENOMORE,"No more results" },
  { WSAECANCELLED,"Call has been canceled" },
  { WSAEINVALIDPROCTABLE,"Procedure call table is invalid" },
  { WSAEINVALIDPROVIDER,"Service provider is invalid" },
  { WSAEPROVIDERFAILEDINIT,"Service provider failed to initialize" },
  { WSASYSCALLFAILURE,"System call failure" },
  { WSASERVICE_NOT_FOUND,"Service not found" },
  { WSATYPE_NOT_FOUND,"Class type not found" },
  { WSA_E_NO_MORE,"No more results" },
  { WSA_E_CANCELLED,"Call was canceled" },
  { WSAEREFUSED,"Database query was refused" },
  { WSAHOST_NOT_FOUND,"Host not found" },
  { WSATRY_AGAIN,"Nonauthoritative host not found" },
  { WSANO_RECOVERY,"This is a nonrecoverable error" },
  { WSANO_DATA,"Valid name,no data record of requested type" },
  { WSA_QOS_RECEIVERS,"QoS receivers" },
  { WSA_QOS_SENDERS,"QoS senders" },
  { WSA_QOS_NO_SENDERS,"No QoS senders" },
  { WSA_QOS_NO_RECEIVERS,"QoS no receivers" },
  { WSA_QOS_REQUEST_CONFIRMED,"QoS request confirmed" },
  { WSA_QOS_ADMISSION_FAILURE,"QoS admission error" },
  { WSA_QOS_POLICY_FAILURE,"QoS policy failure" },
  { WSA_QOS_BAD_STYLE,"QoS bad style" },
  { WSA_QOS_BAD_OBJECT,"QoS bad object" },
  { WSA_QOS_TRAFFIC_CTRL_ERROR,"QoS traffic control error" },
  { WSA_QOS_GENERIC_ERROR,"QoS generic error" },
  { WSA_QOS_ESERVICETYPE,"QoS service type error" },
  { WSA_QOS_EFLOWSPEC,"QoS flowspec error" },
  { WSA_QOS_EPROVSPECBUF,"Invalid QoS provider buffer" },
  { WSA_QOS_EFILTERSTYLE,"Invalid QoS filter style" },
  { WSA_QOS_EFILTERTYPE,"Invalid QoS filter type" },
  { WSA_QOS_EFILTERCOUNT,"Incorrect QoS filter count" },
  { WSA_QOS_EOBJLENGTH,"Invalid QoS object length" },
  { WSA_QOS_EFLOWCOUNT,"Incorrect QoS flow count" },
  { WSA_QOS_EUNKOWNPSOBJ,"Unrecognized QoS object" },
  { WSA_QOS_EPOLICYOBJ,"Invalid QoS policy object" },
  { WSA_QOS_EFLOWDESC,"Invalid QoS flow descriptor" },
  { WSA_QOS_EPSFLOWSPEC,"Invalid QoS provider - specific flowspec" },
  { WSA_QOS_EPSFILTERSPEC,"Invalid QoS provider - specific filterspec" },
  { WSA_QOS_ESDMODEOBJ,"Invalid QoS shape discard mode object" },
  { WSA_QOS_ESHAPERATEOBJ,"Invalid QoS shaping rate object" },
  { WSA_QOS_RESERVED_PETYPE,"Reserved policy QoS element type" }
};