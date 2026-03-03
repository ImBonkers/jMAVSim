package me.drton.jmavsim;

import jssc.SerialPort;
import jssc.SerialPortException;
import me.drton.jmavlib.mavlink.MAVLinkMessage;
import me.drton.jmavlib.mavlink.MAVLinkSchema;
import me.drton.jmavlib.mavlink.MAVLinkStream;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ByteChannel;

/**
 * User: ton Date: 28.11.13 Time: 23:30
 */
public class SerialMAVLinkPort extends MAVLinkPort {
    private MAVLinkSchema schema = null;
    private SerialPort serialPort = null;
    private ByteChannel channel = null;
    private MAVLinkStream stream = null;
    private boolean debug = false;

    // connection information
    String portName;
    int baudRate;
    int dataBits;
    int stopBits;
    int parity;

    // reconnection support
    private boolean autoReconnect = true;
    private long lastReconnectAttempt = 0;
    private static final long RECONNECT_INTERVAL_MS = 500;
    private boolean disconnectReported = false;
    private long lastDataReceived = 0;
    private static final long DATA_TIMEOUT_MS = 2000;

    public SerialMAVLinkPort(MAVLinkSchema schema) {
        super(schema);
        this.schema = schema;
    }

    public void setup(String portName, int baudRate, int dataBits, int stopBits, int parity) {
        this.portName = portName;
        this.baudRate = baudRate;
        this.dataBits = dataBits;
        this.stopBits = stopBits;
        this.parity = parity;
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public void open() throws IOException {
        serialPort = new SerialPort(portName);
        try {
            serialPort.openPort();
            serialPort.setParams(baudRate, dataBits, stopBits, parity);
            serialPort.setDTR(true);
            serialPort.setRTS(true);
        } catch (SerialPortException e) {
            serialPort = null;
            throw new IOException(e);
        }
        channel = new ByteChannel() {
            @Override
            public int read(ByteBuffer buffer) throws IOException {
                try {
                    int available = serialPort.getInputBufferBytesCount();
                    if (available <= 0) {
                        return 0;
                    }

                    byte[] b = serialPort.readBytes(Math.min(available, buffer.remaining()));
                    if (b != null) {
                        buffer.put(b);
                        return b.length;
                    }

                    return 0;

                } catch (SerialPortException e) {
                    throw new IOException(e);
                }
            }

            @Override
            public int write(ByteBuffer buffer) throws IOException {
                try {
                    byte[] b = new byte[buffer.remaining()];
                    buffer.get(b);
                    return serialPort.writeBytes(b) ? b.length : 0;
                } catch (SerialPortException e) {
                    throw new IOException(e);
                }
            }

            @Override
            public boolean isOpen() {
                return serialPort.isOpened();
            }

            @Override
            public void close() throws IOException {
                try {
                    serialPort.closePort();
                } catch (SerialPortException e) {
                    throw new IOException(e);
                }
            }
        };
        stream = new MAVLinkStream(schema, channel);
        stream.setDebug(debug);
    }

    @Override
    public void close() throws IOException {
        if (serialPort == null) {
            return;
        }

        try {
            serialPort.closePort();
        } catch (SerialPortException e) {
            throw new IOException(e);
        }
        serialPort = null;
        stream = null;
    }

    @Override
    public boolean isOpened() {
        return serialPort != null && serialPort.isOpened();
    }

    /**
     * Enable or disable auto-reconnection on disconnect.
     */
    public void setAutoReconnect(boolean autoReconnect) {
        this.autoReconnect = autoReconnect;
    }

    /**
     * Attempt to reconnect to the serial port.
     * @return true if reconnection successful
     */
    private boolean tryReconnect() {
        // Close existing port if any
        if (serialPort != null) {
            try {
                serialPort.closePort();
            } catch (SerialPortException e) {
                // Ignore
            }
            serialPort = null;
            channel = null;
            stream = null;
        }

        // Check if port exists (works for /dev/ttyACM* on Linux)
        java.io.File portFile = new java.io.File(portName);
        if (!portFile.exists()) {
            // Debug: uncomment to see reconnect attempts
            // System.out.println("SerialMAVLinkPort: Port " + portName + " not found, waiting...");
            return false;
        }

        // Try to open
        try {
            open();
            System.out.println("SerialMAVLinkPort: Reconnected to " + portName);
            return true;
        } catch (IOException e) {
            System.out.println("SerialMAVLinkPort: Reconnect failed: " + e.getMessage());
            return false;
        }
    }

    @Override
    public void handleMessage(MAVLinkMessage msg) {
        if (isOpened() && stream != null) {
            try {
                stream.write(msg);
            } catch (Exception e) {
                System.err.println("SerialMAVLinkPort: Write error, triggering reconnect: " + e.getMessage());
                // Close the port - will trigger reconnect on next update
                try {
                    close();
                } catch (IOException ex) {
                    // Ignore
                }
            }
        }
    }

    @Override
    public void update(long t, boolean paused) {
        // Handle reconnection if disconnected
        if (!isOpened()) {
            if (!disconnectReported) {
                System.out.println("SerialMAVLinkPort: Disconnected from " + portName + ", attempting to reconnect...");
                disconnectReported = true;
            }
            if (autoReconnect) {
                long now = System.currentTimeMillis();
                if (now - lastReconnectAttempt > RECONNECT_INTERVAL_MS) {
                    lastReconnectAttempt = now;
                    if (tryReconnect()) {
                        disconnectReported = false;
                    }
                }
            }
            return;
        }

        MAVLinkMessage msg;
        boolean receivedData = false;
        while (isOpened() && stream != null) {
            try {
                msg = stream.read();
                if (msg == null) {
                    break;
                }
                msg.forwarded = true;
                sendMessage(msg);
                receivedData = true;
            } catch (IOException e) {
                System.err.println("SerialMAVLinkPort: Read error, will attempt reconnect: " + e.getMessage());
                // Close the port - will trigger reconnect on next update
                try {
                    close();
                } catch (IOException ex) {
                    // Ignore
                }
                return;
            }
        }

        // Track data reception for timeout detection
        long now = System.currentTimeMillis();
        if (receivedData) {
            lastDataReceived = now;
        } else if (lastDataReceived > 0 && (now - lastDataReceived) > DATA_TIMEOUT_MS) {
            // No data received for too long - force reconnect
            System.out.println("SerialMAVLinkPort: No data received for " + DATA_TIMEOUT_MS + "ms, forcing reconnect");
            try {
                close();
            } catch (IOException ex) {
                // Ignore
            }
            lastDataReceived = 0;
        }
    }

    public void sendRaw(byte[] data) throws IOException {
        try {
            serialPort.writeBytes(data);
        } catch (SerialPortException e) {
            throw new IOException(e);
        }
    }
}
