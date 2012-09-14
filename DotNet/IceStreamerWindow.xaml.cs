// **********************************************************************
//
// Copyright (c) 2003-2011 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************

using System;
using System.Collections.Generic;
using System.Linq; 
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using System.Diagnostics;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;


namespace Ice.wpf.client
{
    /// <summary>
    /// Interaction logic for RayTracerWindows.xaml
    /// </summary>
    public partial class RayTracerWindow : Window
    {

        System.Windows.Threading.DispatcherTimer timer = new System.Windows.Threading.DispatcherTimer();
        private Ice.Communicator communicator_ = null;
        private Streamer.BitmapProviderPrx bitmapProvider_ = null;
        private bool _response = false;
        private RayTracerWindow _window;


        System.Windows.Point previousPoint = new System.Windows.Point(0, 0);
        System.Windows.Point currentPoint = new System.Windows.Point(0, 0);
        static float _depthOfField = 0;
        static float _anglex = 0;
        static float _angley = 0;
        static float _exez = -1000;
        
        public RayTracerWindow()
        {
            InitializeComponent();
            locateOnScreen(this);

            timer.Tick += new EventHandler(dispatcherTimer_Tick); // Everytime timer ticks, timer_Tick will be called
        }

        private void Window_Closed(object sender, EventArgs e)
        {
            if(communicator_ == null)
            {
                return;
            }

            communicator_.destroy();
            communicator_ = null;
        }

        class BitmapProviderCB
        {
            public Ice.AsyncResult _result = null;

            public BitmapProviderCB(RayTracerWindow window)
            {
                _window = window;
            }

            public void getBitmapCB(byte[] bytes)
            {
                lock (this)
                {
                    Debug.Assert(!_response);
                    _response = true;

                    try
                    {
                        MemoryStream stream = new MemoryStream(bytes);
                        BitmapImage b = new BitmapImage();
                        stream.Seek(0, SeekOrigin.Begin);
                        b.BeginInit();
                        b.CacheOption = BitmapCacheOption.OnLoad;
                        b.StreamSource = stream;
                        b.EndInit();
                        
                        _window.bitmapDisplay.Source = b;
                    }
                    catch (System.Exception exMessage)
                    {
                        throw exMessage;
                    }
                }
            }

            public void exception(Exception ex)
            {
                lock (this)
                {
                    Debug.Assert(!_response);
                    _response = true;
                    _window.handleException(ex);
                }
            }

            private bool _response = false;
            private RayTracerWindow _window;
        }

        private void dispatcherTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                if (bitmapProvider_ != null)
                {
                 
                    BitmapProviderCB cb = new BitmapProviderCB(this);
                    cb._result = bitmapProvider_.begin_getBitmap(
                        0, 0, _exez,
                        0, 0, _exez+1000,
                        _anglex, _angley, 0,
                        0, _depthOfField, 0).whenCompleted(cb.getBitmapCB, cb.exception);
                }
            }
            catch (Ice.LocalException ex)
            {
                handleException(ex);
            }
        }

        public void response()
        {
            lock (this)
            {
                Debug.Assert(!_response);
                _response = true;
            }
        }

        public void exception(Exception ex)
        {
            lock (this)
            {
                Debug.Assert(!_response);
                _response = true;
                _window.handleException(ex);
            }
        }
        
        private void handleException(Exception ex)
        {
        }

        static private void locateOnScreen(System.Windows.Window window)
        {
            window.Left = (System.Windows.SystemParameters.PrimaryScreenWidth - window.Width) / 2;
            window.Top = (System.Windows.SystemParameters.PrimaryScreenHeight - window.Height) / 2;
        }
        
        private void bitmapResult_ImageFailed(object sender, ExceptionRoutedEventArgs e)
        {

        }

        private void Window_MouseMove(object sender, MouseEventArgs e)
        {
            //_anglex = 0;
            //_angley = 0;
            int x = (int)e.GetPosition(null).X;
            int y = (int)e.GetPosition(null).Y;
            if (e.MiddleButton == MouseButtonState.Pressed )
            {
                previousPoint.X = x - this.Width / 2;
                previousPoint.Y = y - this.Height / 2;

                _anglex += -(float)(previousPoint.Y - currentPoint.Y) / 10000;
                _angley += (float)(previousPoint.X - currentPoint.X) / 10000;
            }
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                _exez += (float)(previousPoint.Y - y) * 10;
            }
            if (e.RightButton == MouseButtonState.Pressed)
            {
                _depthOfField += (float)(previousPoint.Y - y)*10;
            }
            previousPoint.X = x;
            previousPoint.Y = y;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
            }
            catch (Ice.LocalException ex)
            {
                handleException(ex);
            }
        }

        private void goBtn_Click(object sender, RoutedEventArgs e)
        {
            if (communicator_ == null)
            {
                Ice.InitializationData initData = new Ice.InitializationData();
                initData.properties = Ice.Util.createProperties();
                initData.dispatcher = delegate(System.Action action, Ice.Connection connection)
                {
                    Dispatcher.BeginInvoke(DispatcherPriority.Normal, action);
                };
                communicator_ = Ice.Util.initialize(initData);
            }
            if (bitmapProvider_ == null)
            {
                string connection = "IceStreamer:tcp -h " + tbIpAddress.Text + " -p 10000 -z";
                Ice.ObjectPrx prx = communicator_.stringToProxy(connection);
                prx = prx.ice_timeout(1000);
                bitmapProvider_ = Streamer.BitmapProviderPrxHelper.uncheckedCast(prx);
            }

            timer.Interval = new TimeSpan(1000);                   // Timer will tick event/second
            timer.Start();                                        // Start the timer
        }
    }
}
