import React, { useEffect, useState, useRef } from 'react';
import { InputGroup, FormControl, Form } from 'react-bootstrap';


/**
 * checks if the addr is a valid ethereum address (hex and 42 char long including the 0x) 
 * @param {string} addr 
 */
function isValid(addr) {
    const re = /[0-9A-Fa-f]{40}/g;
    addr = addr.trim();
    const ans = (
        (addr.substr(0, 2) === '0x' && re.test(addr.substr(2))) //with the 0x
        || (re.test(addr)));  //without the 0x
    return ans;
}

export default function AddressSearchBar({onSubmit, inputAddress, setInputAddress, setInitialized}) {
    const inputEl = useRef(null);
    const [invalid, setInvalid] = useState(false);

    useEffect(() => {
        if (inputAddress.length > 0) {
            inputEl.current.value = inputAddress;
        }
    }, [inputAddress])

    const submitInputAddress = addr => {
        if (!isValid(addr)) {
            setInvalid(true);
            return;
        }
        setInvalid(false);
        onSubmit(addr);
    }

    const onChangeInputAddress = e => {
        const addr = e.target.value;
        e.preventDefault();
        setInputAddress(addr);
        setInitialized(false);
    }

    return (
        <div className="autocomplete-input-box">
            <InputGroup 
                onSubmit={submitInputAddress} 
                className="fixed-width col-md-12 autocomplete-input__container" 
                hasValidation
            >
                <FormControl onKeyPress={(e) => {
                    if (e.key !== 'Enter') return;
                    e.preventDefault();
                    submitInputAddress(inputEl.current.value);
                }}
                    onChange={onChangeInputAddress}
                    placeholder='0x0000000000000000000000000000000000000000'
                    className="search-bar autocomplete-input"
                    isInvalid={invalid}
                    ref={inputEl}
                    area-describedby='addressSearchHelp'
                >
                </FormControl>

                <Form.Control.Feedback type="invalid">
                    Please enter a valid ethereum address.
                </Form.Control.Feedback>
            </InputGroup>
            <div className="autocomplete-input-box__footer">
                <div>
                    <b>Instructions</b> &nbsp; Enter an ethereum smart contract address, and press enter.
                </div>
                <div>
                    <b>Disclaimer</b> &nbsp; This is a proof-of-concept demo. It's predictions may be inaccurate. The underlying model was trained on limited data sizes that restrict its generalization capabilities. Use at your own discretion.
                </div>
            </div>
        </div>
    )
}