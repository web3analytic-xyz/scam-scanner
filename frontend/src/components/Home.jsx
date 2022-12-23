import React, { useState } from 'react';
import Header from './Header';
import Footer from './Footer';
import AddressSearchBar from './AddressSearchBar';
import ResultsPage from './ResultsPage';
import axios from 'axios';

export default function HomePage() {
    const [inputAddress, setInputAddress] = useState('');
    const [prediction, setPrediction] = useState(-1);
    const [probability, setProbability] = useState(-1);
    const [initialized, setInitialized] = useState(false);
    const [error, setError] = useState(false);

    // What to do in an actual query
    const onAddressSubmit = (addr) => {
        axios.get('http://127.0.0.1:8000/api/scan/' + addr).then(response => {
            if (response.status === 200) {
                if (response.data.success) {
                    setPrediction(response.data.pred);
                    setProbability(response.data.prob);
                } else {
                    setError(true);
                }
            } else {
                setError(true);
            }
            setInitialized(true);
        });
    };

    return (
        <div className="container">
            <div className="row">
                <Header />
                <div className="full-page main col-12">
                    <div className="justify-center row">
                        <AddressSearchBar 
                            onSubmit={onAddressSubmit}
                            inputAddress={inputAddress}
                            setInputAddress={setInputAddress} 
                            setInitialized={setInitialized}
                        />
                    </div>
                    <div className="justify-center row">
                        <ResultsPage 
                            address={inputAddress}
                            prediction={prediction}
                            probability={probability}
                            error={error}
                            initialized={initialized}
                        />
                    </div>
                </div>
                <Footer />
            </div>
        </div>
    );
}