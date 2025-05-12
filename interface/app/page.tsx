'use client';

import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen p-8">
      <main className="max-w-6xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Kangaroo ECDLP Solver</h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Powerful solution for tackling the Elliptic Curve Discrete Logarithm Problem
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h2 className="text-2xl font-bold mb-4">Puzzle #135</h2>
            <p className="mb-6 text-gray-600 dark:text-gray-300">
              Join forces with other contributors to solve ECDLP problems more efficiently. Track your contributions and see real-time statistics.
            </p>
            <div className="space-y-4">
              <Link 
                href="/pool" 
                className="block w-full bg-blue-500 hover:bg-blue-600 text-white text-center py-2 px-4 rounded transition"
              >
                Pool Dashboard
              </Link>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h2 className="text-2xl font-bold mb-4">Features</h2>
            <ul className="space-y-3">
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>254-bit search range using fixed-size 256-bit arithmetic</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>Endomorphism optimization comparing only Y-coordinates</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>Fast modular inversion using Delayed Right Shift 62 bits</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>Multi-GPU support with CUDA optimization</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                <span>Pool mode with Bitcoin address identification</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold mb-4">How to Get Started</h2>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-xl font-semibold mb-2">Clone the Repository</h3>
              <div className="bg-gray-100 dark:bg-gray-900 p-4 rounded">
                <code className="text-sm">
                  git clone https://github.com/GiladLeef/kangaroo/
                </code>
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold mb-2">Compile on Ubuntu 24.04 x86_64 Machine</h3>
              <div className="bg-gray-100 dark:bg-gray-900 p-4 rounded">
                <pre className="text-sm whitespace-pre-wrap">
                  # Install a CUDA driver<br/>
                  sudo apt-get install -y nvidia-open<br/><br/>
                  
                  # Install Nvidia CUDA Toolkit<br/>
                  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb<br/>
                  sudo dpkg -i cuda-keyring_1.1-1_all.deb<br/>
                  sudo apt-get update<br/>
                  sudo apt-get -y install cuda-toolkit-12-8<br/><br/>
                  
                  # Install g++ and make<br/>
                  sudo apt-get install make g++<br/><br/>
                  
                  # Compile CPU-ONLY version<br/>
                  make<br/><br/>
                  
                  # Compile with GPU support:<br/>
                  make gpu=1 ccap=&lt;integer&gt; all
                </pre>
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold mb-2">Connecting to Our Pool</h3>
              <div className="bg-gray-100 dark:bg-gray-900 p-4 rounded">
                <code className="text-sm">
                  ./kangaroo -c server_ip -p -a YOUR_BITCOIN_ADDRESS [other options]
                </code>
              </div>
              <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                Replace YOUR_BITCOIN_ADDRESS with your Bitcoin P2PK address to receive credit for your contributions.
              </p>
            </div>
          </div>
        </div>
      </main>

      <footer className="max-w-6xl mx-auto mt-12 pt-6 border-t border-gray-200 dark:border-gray-700 text-center text-gray-500 dark:text-gray-400 text-sm">
        <p>Kangaroo ECDLP Solver - Pollard's Kangaroo for SECP256K1</p>
      </footer>
    </div>
  );
}
