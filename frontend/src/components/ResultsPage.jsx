import React from 'react';

export default function ResultsPage({
    address,
    prediction,
    probability,
    error,
    initialized
}) {
    if (!initialized) {
        return <div></div>;
    }

    var content;
    if (error) {
        content = (
            <div className="results-content">
                Something went wrong! Check that the address is for a smart contract (not an externally-owned account).
            </div>
        )
    } else {
        let etherscan = "https://etherscan.io/address/" + address;
        probability = (probability * 100).toFixed(1);
        var predictionText;
        if (prediction) {
            predictionText = <span style={{color: "rgb(243, 66, 66)"}}>YES</span>;
        } else {
            probability = 100 - probability;
            predictionText = <span style={{color: "rgb(3,125,80)"}}>NO</span>;
        }
        content = (
            <div className="results-content">
                <div>
                    Processed smart contract at address <a href={etherscan}>{address}</a>. 
                </div>
                <div className="container">
                    <div className="justify-center row">
                        <div className="col-sm">
                            <div className="size-container">
                                <h3>Malicious?</h3>
                                <div className="stat-container__value-container">
                                    <div className="stat-container__value-wrap">
                                        <div className="stat-container__value size">{predictionText}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="col-sm">
                            <div className="size-container">
                                <h3>Confidence</h3>
                                <div className="stat-container__value-container">
                                    <div className="stat-container__value-wrap">
                                        <div className="stat-container__value size">{probability}</div>
                                    </div>
                                    <div className="stat-container__unit">
                                        %
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        )
    }
    return (
        <div className="results-page">
            <div> 
                <span className="logo-results">Scan Results</span>
            </div>
            {content}
        </div>
    )
}