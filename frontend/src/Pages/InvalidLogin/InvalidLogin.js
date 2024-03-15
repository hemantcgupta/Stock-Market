import React, { Component } from 'react';
import config from '../../Constants/config';
import Button from '@material-ui/core/Button';
import { images } from '../../Constants/images';
import './InvalidLogin.scss'

class InvalidLogin extends Component {
    constructor(props) {
        super(props);
        this.state = {
            loading: false
        }
    }

    redirect = () => {
        this.setState({ loading: true })
        window.location.href = `${config.API_URL}/initiatelogin`
    }
    render() {
        return (
            <div className='invalidLogin'>
                <h1>Oops..!! Invalid Login</h1>
                <Button
                    type={`submit`}
                    fullWidth
                    variant="contained"
                    color="primary"
                    className='invalid-btn'
                    onClick={this.redirect}>
                    {this.state.loading ? <img src={images.ellipsis_loader} alt='' /> : 'Click here to try again'}
                </Button>
            </div>
        );
    }

}

export default InvalidLogin;