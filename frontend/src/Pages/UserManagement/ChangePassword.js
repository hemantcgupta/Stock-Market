import React, { Component } from 'react';
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import api from '../../API/api'
import $ from 'jquery';
import './UserManagement.scss';
import _ from 'lodash';
import Swal from 'sweetalert2';




class ChangePassword extends Component {
    constructor(props) {
        super(props);
        this.state = {
            msg: "",
            error: '',
            validate: false,

        };

    }

    componentDidMount() {
    }

    _validate = (event) => {
        this.setState({ validate: false })

        let ChangeOldpassword = this.refs.ChangeOldpassword.value.trim()
        let ChangeNewpassword = this.refs.ChangeNewpassword.value.trim()
        let ChangeconfirmNewPassword = this.refs.ChangeconfirmNewPassword.value.trim()

        if (ChangeOldpassword === '') {
            this.setState({
                error: 'Old Password is required field'
            })
        } else if (ChangeOldpassword.length < 8) {
            this.setState({
                error: 'Password should be min 8 characters long'
            })
        } else if (ChangeNewpassword === '') {
            this.setState({
                error: 'New Password is required field'
            })
        } else if (ChangeNewpassword.length < 8) {
            this.setState({
                error: 'Password should be min 8 characters long'
            })
        } else if (ChangeNewpassword !== ChangeconfirmNewPassword) {
            this.setState({
                error: 'Confirm Password did Match with New Password'
            })
        } else {
            this.setState({
                error: '',
                validate: true
            })
        }
    }

    changePassword = () => {
        const oldPassword = (this.refs.ChangeOldpassword.value).trim();
        const password = (this.refs.ChangeNewpassword.value).trim();
        let data = {
            old_password: oldPassword,
            new_password: password
        }

        api.post('api/passwordchange', data)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `'Password changed successfully`,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.reset()
                })
            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: err.response.data.old_password[0],
                    icon: 'error',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.reset()
                })
            })
    }

    reset() {
        $('input').val('');
    }



    render() {
        let errorMgs = this.state.error !== '' ? <p>{this.state.error}</p> : <span />
        return (
            <div>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className='user-module'>
                    <div className="clearfix"></div>
                    <div className="row">
                        <div className="col-md-12 col-sm-12 col-xs-12">
                            <div className="x_panel fade">
                                <div className="add">
                                    <h2>Change Password</h2>
                                    <div className="clearfix"></div>
                                </div>
                                <div>
                                    <div className="module-form ">
                                        <div className="form-group">
                                            <label htmlFor="password">Old Password * :</label>
                                            <input type="password" ref="ChangeOldpassword" className="form-control" name="ChangeOldpassword" onChange={() => this._validate()} />
                                        </div>

                                        <div className="form-group">
                                            <label htmlFor="password">New Password * :</label>
                                            <input type="password" ref="ChangeNewpassword" className="form-control" name="ChangeNewpassword" id="ChangeNewpassword" onChange={() => this._validate()} />
                                        </div>

                                        <div className="form-group">
                                            <label htmlFor="changeconfirmPassword">Confirm New Password * :</label>
                                            <input type="password" ref="ChangeconfirmNewPassword" className="form-control" name="changeconfirmPassword" onChange={() => this._validate()} />
                                        </div>

                                        {errorMgs}
                                    </div>
                                    <div className='action-btn'>
                                        <button type="button" className="btn" onClick={this.changePassword.bind(this)} disabled={!this.state.validate}>Change Password</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default ChangePassword;