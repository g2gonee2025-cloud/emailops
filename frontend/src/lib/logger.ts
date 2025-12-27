
export const LogLevel = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
} as const;

export type LogLevel = typeof LogLevel[keyof typeof LogLevel];

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  data?: unknown;
}

const LevelNames = {
  [LogLevel.DEBUG]: 'DEBUG',
  [LogLevel.INFO]: 'INFO',
  [LogLevel.WARN]: 'WARN',
  [LogLevel.ERROR]: 'ERROR',
};

class Logger {
  private logs: LogEntry[] = [];
  private maxLogs = 1000;
  private level: LogLevel = LogLevel.INFO;

  constructor() {
    // Default to DEBUG in dev, INFO in prod
    if (import.meta.env.DEV) {
      this.level = LogLevel.DEBUG;
    }
  }

  public setLevel(level: LogLevel) {
    this.level = level;
  }

  public getLogs(): LogEntry[] {
    return this.logs;
  }

  private addLog(level: LogLevel, message: string, data?: unknown) {
    if (level < this.level) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      data,
    };

    this.logs.push(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    this.consolePrint(entry);
  }

  private consolePrint(entry: LogEntry) {
    const style = this.getConsoleStyle(entry.level);
    const prefix = `[${LevelNames[entry.level]}]`;

    if (entry.data) {
        console.log(`%c${prefix} ${entry.message}`, style, entry.data);
    } else {
        console.log(`%c${prefix} ${entry.message}`, style);
    }
  }

  private getConsoleStyle(level: LogLevel): string {
    switch (level) {
      case LogLevel.DEBUG: return 'color: #94a3b8; font-weight: bold;';
      case LogLevel.INFO: return 'color: #3b82f6; font-weight: bold;';
      case LogLevel.WARN: return 'color: #f59e0b; font-weight: bold;';
      case LogLevel.ERROR: return 'color: #ef4444; font-weight: bold;';
      default: return 'color: inherit;';
    }
  }

  public debug(message: string, data?: unknown) {
    this.addLog(LogLevel.DEBUG, message, data);
  }

  public info(message: string, data?: unknown) {
    this.addLog(LogLevel.INFO, message, data);
  }

  public warn(message: string, data?: unknown) {
    this.addLog(LogLevel.WARN, message, data);
  }

  public error(message: string, error?: unknown) {
    this.addLog(LogLevel.ERROR, message, error);
  }
}

export const logger = new Logger();
