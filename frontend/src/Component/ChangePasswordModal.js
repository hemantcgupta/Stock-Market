import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import axios from 'axios';
import AlertCard from './AlertCard';
import Validator from '../Constants/validator';
import cookieStorage from "../Constants/cookie-storage";
import _ from 'lodash';
import { images } from '../Constants/images';
import config from '../Constants/config';
import Swal from 'sweetalert2';
import "./component.scss";


class ChangePasswordModal extends Component {
    constructor(props) {
        super(props);
        this.state = {
            newPass: '',
            oldPass: '',
            oldPassMessage: '',
            newPassMessage: '',
            loading:false,
            disable:true
        }
    }

    componentWillReceiveProps(props) {
        if (props) {
            this.setState({ displayModal: props.displayModal })
        }
    }

    handleChange = (e) => {
        e.stopPropagation();
        this.setState({ [e.target.name]: e.target.value }, this._validate(e))
    }

    _validate = (e) => {
        let name = e.target.name;
        let value = e.target.value;
        let msg = '';
        if (name === 'oldPass') {
            let isValid = Validator.validatePassword(value);
            msg = value === '' ? `Old password should not be empty*` : ``;
            this.setState({ oldPassMessage: msg });
        } else if (name === 'newPass') {
            let isValid = Validator.validatePassword(value);
            msg = value === '' ? `New password should not be empty*` : ``;
            this.setState({ newPassMessage: msg });
        }
        setTimeout(() => {
            this.formValid();
        }, 1000);
    }

    formValid = () => {
        const { oldPassMessage, newPassMessage, oldPass, newPass } = this.state;
        const errorCheck = _.isEmpty(oldPassMessage && newPassMessage)
        const emptyDataCheck = !_.isEmpty(oldPass && newPass)
        if (errorCheck && emptyDataCheck) {
            this.setState({ disable: false })
        } else {
            this.setState({ disable: true })
        }
    }

    changePassword = () => {
        if (!this.state.loading) {
            if (!this.state.disable) {
                this.setState({ loading: true })
                const token = cookieStorage.getCookie('Authorization')
                axios.post(`${config.API_URL}/api/passwordchange`, {
                    old_password: this.state.oldPass,
                    new_password: this.state.newPass
                }, {
                    headers: {
                        'Authorization': token,
                        'Content-Type': 'application/json',
                    },
                })
                    .then((response) => {
                        this.setState({ loading: false });
                        Swal.fire({
                            title: 'Done!',
                            text: response.data.message,
                            icon: 'success',
                            confirmButtonText: 'Ok'
                        }).then(() => {
                            this.getUpdatedResetPassStatus();
                            this.setState({ displayModal: false })
                        })
                    })
                    .catch((error) => {
                        console.log('error', error)
                        this.setState({ loading: false })
                        if (error.response !== undefined) {
                            Swal.fire({
                                title: 'Error!',
                                text: error.response.data.old_password,
                                icon: 'error',
                                confirmButtonText: 'Ok'
                            }).then(() => {
                                this.resetState()
                            })
                        } else {
                            Swal.fire({
                                title: 'Error!',
                                text: 'Network Error',
                                icon: 'error',
                                confirmButtonText: 'Ok'
                            }).then(() => {
                                this.resetState()
                            })
                        }
                    });
            }
        }
    }

    getUpdatedResetPassStatus = () => {
        const token = cookieStorage.getCookie('Authorization')
        axios.get(`${config.API_URL}/api/user`, {
            headers: {
                'Authorization': token,
                'Content-Type': 'application/json',
            },
        })
            .then((response) => {
                cookieStorage.createCookie('userDetails', JSON.stringify(response.data), 1)
            })
            .catch((error) => {
                console.log('error', error)
            });
    }

    resetState = () => {
        this.setState({
            newPass: '',
            oldPass: '',
            oldPassMessage: '',
            newPassMessage: '',
            disable: true,
        })
    }

    render() {
        const { alertData } = this.props;
        return (
            <div>
                <Modal
                    show={this.state.displayModal}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header >
                        <Modal.Title id='ModalHeader'>Change your password</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <label >Old Password * :</label>
                        <input type="password" ref="oldPass" className="form-control" name="oldPass" value={this.state.oldPass} onChange={(e) => this.handleChange(e)} />
                        <p>{this.state.oldPassMessage}</p>

                        <label>New Password* :</label>
                        <input type="password" ref="newPass" className="form-control" name="newPass" value={this.state.newPass} onChange={(e) => this.handleChange(e)} />
                        <p>{this.state.newPassMessage}</p>

                    </Modal.Body>
                    <Modal.Footer>
                        <button disabled={this.state.disable} className='btn btn-primary change-password' onClick={() => this.changePassword()}>
                            {this.state.loading ? <img src={images.ellipsis_loader} alt='' /> : `Change Password`}
                        </button>
                    </Modal.Footer>
                </Modal>
            </div >
        );
    }
}
export default ChangePasswordModal;




