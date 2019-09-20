#include "ActivityMonitor.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <time.h>

void ActivityMonitor::reset_state() {
  backend()->reset_state();
}

SPIKE_MAKE_STUB_INIT_BACKEND(ActivityMonitor);
