import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import Access from "../../Constants/accessValidation";
import String from "../../Constants/validator";
import api from '../../API/api'
import Modal from 'react-bootstrap-modal';
import './UserManagement.scss';
import _ from 'lodash';
import Swal from 'sweetalert2';
import cookieStorage from '../../Constants/cookie-storage';


let routeAccess = {
    selectedRouteGroup: ['Domestic']
};

class EditUser extends Component {
    constructor(props) {
        super(props);
        this.state = {
            details: null,
            msg: "",
            regions: [],
            regionSelected: 'Null',
            regionSelectedID: 'Null',
            countries: [],
            countrySelected: 'Null',
            countrySelectedID: 'Null',
            cities: [],
            citySelected: 'Null',
            routeGroup: 'Domestic',
            routeRegions: [],
            routeCountries: [],
            routes: [],
            selectedRouteRegion: 'Null',
            selectedRouteCountry: 'Null',
            selectedRoute: 'Null',
            error: '',
            errorPass: '',
            posNetwork: false,
            routeNetwork: false,
            posAccess: '',
            validate: false,
            disable: false,
            checkboxArray: [],
            checkboxSelctedCountIndex: [],
            displayModal: false

        };
        this.hideAlertSuccess = this.hideAlertSuccess.bind(this);

    }

    componentDidMount() {
        if (Access.accessValidation('Users', 'editUser')) {
            const regions = JSON.parse(cookieStorage.getCookie('Regions'));
            this.setState({ regions: regions });
            this.getUserDetails();
            this.getSystemModulelist();
        } else {
            this.props.history.push('/404')
        }
    }

    callRegion = () => {
        const e = document.getElementById('region')
        const regionSelected = e.options[e.selectedIndex].value;
        this.setState({
            regionSelected: `'${regionSelected}'`,
            posAccess: regionSelected === '*' ? '' : `#'${regionSelected}'#*`,
            countries: [],
            cities: [],
            error: ''
        }, () => { this.getCountries() });
    }

    callCountry = () => {
        const { regionSelected } = this.state;
        const e = document.getElementById('country')
        const countrySelected = e.options[e.selectedIndex].value;
        this.setState({
            countrySelected: `'${countrySelected}'`,
            posAccess: countrySelected === '*' ? `#${regionSelected}#*` : `#${regionSelected}#'${countrySelected}'#*`,
            cities: [],
            error: ''
        }, () => { this.getCities() });
    }

    callCity = (e) => {
        const { regionSelected, countrySelected } = this.state;
        const citySelected = e.target.value;
        this.setState({
            citySelected: `'${citySelected}'`,
            posAccess: citySelected === '*' ? `#${regionSelected}#${countrySelected}#*` : `#${regionSelected}#${countrySelected}#'${citySelected}'#*`,
            error: ''
        })
    }

    getUserDetails = () => {
        const id = this.props.location.state ? this.props.location.state.data.id : ''
        api.get(`rest/users/${id}/`)
            .then((response) => {
                if (response) {
                    const userData = response.data;
                    const id = (userData.systemmodule).split(',').map((d) => parseInt(d));
                    this.setState({
                        details: userData,
                        checkboxSelctedCountIndex: id
                    });
                    this.getPOSAccessStatus(userData);
                    this.getRouteAccessStatus(userData);
                }
            })
            .catch((err) => {
                console.log("something went wrong", err);
            })
    }

    getCountries = () => {
        if (this.state.regionSelected !== '*') {
            api.get(`getCountryByRegionId?regionId=${encodeURIComponent(this.state.regionSelected)}`)
                .then((response) => {
                    const countries = response.data;
                    this.setState({
                        countries: countries,
                    }, () => this.getCities());
                })
                .catch((err) => {
                    console.log("something went wrong with countries Api");
                })
        }
    }

    getCities = () => {
        if (this.state.countrySelected !== '*') {
            api.get(`getCityByCountryCode?countryCode=${this.state.countrySelected}`)
                .then((response) => {
                    const cities = response.data;
                    this.setState({
                        cities: cities,
                    });
                })
                .catch((err) => {
                    console.log("something went wrong with countries Api");
                })
        }
    }

    callRouteGroup(e) {
        const routeGroup = e.target.value;
        routeAccess['selectedRouteGroup'] = [routeGroup]
        delete routeAccess.selectedRouteRegion
        delete routeAccess.selectedRouteCountry
        delete routeAccess.selectedRoute
        this.setState({ routeGroup, routeRegions: [], routeCountries: [], routes: [] }, () => this.getRouteRegions())
    }

    callRouteRegion = () => {
        const e = document.getElementById('routeRegion')
        const regionSelected = e.options[e.selectedIndex].value;
        if (regionSelected !== '*') {
            routeAccess['selectedRouteRegion'] = [regionSelected]
        } else {
            delete routeAccess.selectedRouteRegion
        }
        delete routeAccess.selectedRouteCountry
        delete routeAccess.selectedRoute
        this.setState({
            selectedRouteRegion: `'${regionSelected}'`,
            routeCountries: [],
            routes: [],
            error: ''
        }, () => { this.getRouteCountries() });
    }

    callRouteCountry = () => {
        const e = document.getElementById('routeCountry')
        const countrySelected = e.options[e.selectedIndex].value;
        if (countrySelected !== '*') {
            routeAccess['selectedRouteCountry'] = [countrySelected]
        } else {
            delete routeAccess.selectedRouteCountry
        }
        delete routeAccess.selectedRoute
        this.setState({
            selectedRouteCountry: `'${countrySelected}'`,
            routes: [],
            error: ''
        }, () => { this.getRoutes() });
    }

    callRoute = (e) => {
        if (e.target.value !== '*') {
            routeAccess['selectedRoute'] = [e.target.value]
        } else {
            delete routeAccess.selectedRoute
        }
        this.setState({
            error: ''
        })
    }

    getRouteRegions = () => {
        api.get(`getRouteRegion?routeGroup='${this.state.routeGroup}'`)
            .then((response) => {
                const regions = response.data.response;
                this.setState({
                    routeRegions: regions,
                }, () => this.getRouteCountries());
            })
            .catch((err) => {
                console.log("something went wrong with countries Api");
            })
    }

    getRouteCountries = () => {
        if (this.state.selectedRouteRegion !== '*') {
            api.get(`getRouteCountryByRegionId?routeGroup='${this.state.routeGroup}'&regionId=${this.state.selectedRouteRegion}`)
                .then((response) => {
                    const countries = response.data.response;
                    this.setState({
                        routeCountries: countries,
                    }, () => this.getRoutes());
                })
                .catch((err) => {
                    console.log("something went wrong with countries Api");
                })
        }
    }

    getRoutes = () => {
        if (this.state.selectedRouteCountry !== '*') {
            api.get(`getRouteCityByCountryCode?routeGroup='${this.state.routeGroup}'&countryId=${this.state.selectedRouteCountry}`)
                .then((response) => {
                    const routes = response.data.response;
                    this.setState({
                        routes: routes,
                    });
                })
                .catch((err) => {
                    console.log("something went wrong with countries Api");
                })
        }
    }

    getSystemModulelist = () => {
        api.get(`getsysmod`)
            .then((response) => {
                if (response) {
                    if (response.data.response.length > 0) {
                        this.setState({ checkboxArray: response.data.response });
                    }
                }
            })
            .catch((err) => {
            })
    }

    _validate = (event) => {
        this.setState({ validate: false })
        if (this.refs.name.value === '') {
            this.setState({
                error: 'Name is required field'
            })
        } else if (this.refs.role.value === '') {
            this.setState({
                error: 'Role is Required field'
            })
        } else if (this.state.posAccess === '') {
            this.setState({
                error: 'Please select pos access'
            })
        } else if (this.state.checkboxSelctedCountIndex.length === 0) {
            this.setState({
                error: 'Please select system module'
            })
        } else {
            this.setState({
                error: '',
                validate: true
            }, () => this.updateUser())
        }
    }

    updateUser = () => {
        if (this.state.validate) {
            let systemModules = this.state.checkboxSelctedCountIndex;
            if (systemModules.length > 0) {
                systemModules = systemModules.filter((value) => {
                    return !Number.isNaN(value);
                });
                systemModules = systemModules.join(',');
            } else {
                systemModules = '';
            }
            const name = this.refs.name.value;
            const role = this.refs.role.value;
            const access = this.state.posAccess;
            const route_access = JSON.stringify(routeAccess);
            var data = {
                username: name,
                email: this.state.details.email,
                role: role,
                access: access,
                route_access: route_access,
                password: this.state.details.password,
                systemmodule: systemModules
            };

            api.put(`rest/users/${this.props.location.state.data.id}/`, data)
                .then((res) => {
                    Swal.fire({
                        title: 'Updated!',
                        text: `Profile updated successfully`,
                        icon: 'success',
                        confirmButtonText: 'Ok'
                    }).then(() => {
                        this.hideAlertSuccess()
                    })

                })
                .catch((err) => {
                    Swal.fire({
                        title: 'Error!',
                        text: 'Unable to update profile',
                        icon: 'error',
                        confirmButtonText: 'Ok'
                    })
                })
        }
    }


    hideAlertSuccess() {
        this.props.history.push(`/searchUser`)
    }

    onPOSNetworkChange() {
        this.setState({ posNetwork: !this.state.posNetwork, error: '' }, () => this.setPOSNetworkLevel())
    }

    setPOSNetworkLevel() {
        if (this.state.posNetwork) {
            this.setState({ posAccess: '#*', regions: [], countries: [], cities: [] })
        }
        else {
            const regions = JSON.parse(cookieStorage.getCookie('Regions'));
            this.setState({ regions: regions });
            this.setState({ posAccess: '' })
        }
    }

    onRouteNetworkChange() {
        this.setState({ routeNetwork: !this.state.routeNetwork, error: '' }, () => this.setRouteNetworkLevel())
    }

    setRouteNetworkLevel() {
        if (this.state.routeNetwork) {
            routeAccess = {}
            this.setState({ routeRegions: [], routeCountries: [], routes: [], selectedRouteRegion: '', selectedRouteCountry: '', selectedRoute: '' })
        }
        else {
            this.setState({ routeGroup: 'Domestic' }, () => this.getRouteRegions())
            routeAccess = { selectedRouteGroup: ['Domestic'] }
        }
    }

    getPOSAccessStatus(details) {
        if (details.access === '#*') {
            this.setState({ posNetwork: true, posAccess: '#*' })
        } else {
            let access = details.access;
            let accessList = access.split('#');
            this.setState({
                regionSelected: accessList[1],
                countrySelected: accessList[2],
                citySelected: accessList[3],
                posNetwork: false
            }, () => this.getCountries())
            if (accessList[2] === '*') {
                this.setState({
                    posAccess: `#${accessList[1]}#${accessList[2]}`,
                })
            } else {
                this.setState({
                    posAccess: `#${accessList[1]}#${accessList[2]}#${accessList[3]}`,
                })
            }
        }

    }

    getRouteAccessStatus(details) {
        const access = details.route_access;
        if (Object.keys(access).length === 0) {
            this.setState({ routeNetwork: true })
            routeAccess = {}
        } else {
            if ((access).hasOwnProperty('selectedRouteGroup')) {
                routeAccess['selectedRouteGroup'] = access['selectedRouteGroup']
                this.setState({ routeGroup: access['selectedRouteGroup'].join(',') })
            }
            if ((access).hasOwnProperty('selectedRouteRegion')) {
                routeAccess['selectedRouteRegion'] = access['selectedRouteRegion']
                this.setState({ selectedRouteRegion: `'${access['selectedRouteRegion'].join(',')}'` })
            }
            if ((access).hasOwnProperty('selectedRouteCountry')) {
                routeAccess['selectedRouteCountry'] = access['selectedRouteCountry']
                this.setState({ selectedRouteCountry: `'${access['selectedRouteCountry'].join(',')}'` })
            }
            if ((access).hasOwnProperty('selectedRoute')) {
                routeAccess['selectedRoute'] = access['selectedRoute']
                this.setState({ selectedRoute: `'${access['selectedRoute'].join(',')}'` })
            }
        }
        setTimeout(() => {
            this.getRouteRegions()
        }, 1000);

    }

    onClickCheckbox(index) {
        const checkboxSelctedCountIndex = this.state.checkboxSelctedCountIndex;
        if (checkboxSelctedCountIndex.includes(index)) {
            checkboxSelctedCountIndex.splice(checkboxSelctedCountIndex.findIndex(value => value === index), 1);
        } else {
            checkboxSelctedCountIndex.push(index);
        }
        this.setState({
            checkboxSelctedCountIndex,
            error: ''
        });
    }

    _validatePass = (event) => {
        this.setState({ disable: false })
        let password = this.refs.password.value.trim();
        let confirmPassword = this.refs.confirmPassword.value.trim()
        if (password === '') {
            this.setState({
                errorPass: 'Password is required field'
            })
        }
        else if (password.length < 8) {
            this.setState({
                errorPass: 'Password should be min 8 characters long'
            })
        }
        else if (password !== confirmPassword) {
            this.setState({
                errorPass: 'Confirm Password should Match with Password'
            })
        }
        else {
            this.setState({
                errorPass: '',
                disable: true
            })
        }
    }

    updateUserPassword(e) {
        if (this.state.disable) {
            const myDetails = this.state.details;
            const password = this.refs.password.value.trim();

            let data = {
                username: myDetails.username,
                email: myDetails.email,
                role: myDetails.role,
                access: myDetails.access,
                route_access: JSON.stringify(myDetails.route_access),
                password: password,
                is_resetpwd: 'True'
            }

            api.put(`rest/users/${this.props.location.state.data.id}/`, data)
                .then((res) => {
                    Swal.fire({
                        title: 'Updated!',
                        text: `Password Updated successfully`,
                        icon: 'success',
                        confirmButtonText: 'Ok'
                    }).then(() => {
                        this.hidePopup()
                    })

                })
                .catch((err) => {
                    Swal.fire({
                        title: 'Error',
                        text: `Something went wrong`,
                        icon: 'error',
                        confirmButtonText: 'Ok'
                    }).then(() => {
                        this.refs.password.value = "";
                        this.refs.confirmPassword.value = "";
                    })
                });
        }
    }

    hidePopup = () => {
        this.setState({ displayModal: false, disable: false })
        this.refs.password.value = "";
        this.refs.confirmPassword.value = "";
    }

    render() {
        let errorMgs = this.state.error !== '' ? <p>{this.state.error}</p> : <span />
        let errorMgsPass = this.state.errorPass !== '' ? <p style={{ color: 'red' }}>{this.state.errorPass}</p> : <span />

        let regionOptionItems = this.state.regions.map((region) =>
            <option value={region.Region} selected={String.removeQuotes(this.state.regionSelected) === region.Region}>{region.Region}</option>
        );
        let countryOptionItems = this.state.countries.map((country) =>
            <option value={country.CountryCode} selected={String.removeQuotes(this.state.countrySelected) === country.CountryCode}>{country.CountryCode}</option>
        );
        let cityOptionItems = this.state.cities.map((city) =>
            <option value={city.CityCode} selected={String.removeQuotes(this.state.citySelected) === city.CityCode}>{city.CityCode}</option>
        );
        let routeRegions = this.state.routeRegions.map((region) =>
            <option value={region.Route_Region} selected={String.removeQuotes(this.state.selectedRouteRegion) === region.Route_Region}>{region.Route_Region}</option>
        );
        let routeCountries = this.state.routeCountries.map((country) =>
            <option value={country.Route_Country} selected={String.removeQuotes(this.state.selectedRouteCountry) === country.Route_Country}>{country.Route_Country}</option>
        );
        let routes = this.state.routes.map((route) =>
            <option value={route.Route} selected={String.removeQuotes(this.state.selectedRoute) === route.Route}>{route.Route}</option>
        );

        let details = this.state.details;
        return (
            <div>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className='user-module'>
                    <div className="clearfix"></div>
                    <div className="row">
                        <div className="col-md-12 col-sm-12 col-xs-12">
                            <div className="x_panel fade user-module-main">
                                <div className="add">
                                    <h2>Edit User</h2>
                                    {/* <button type="button" className="btn btn-primary reset" onClick={() => this.setState({ displayModal: true })}>RESET PASSWORD</button> */}
                                </div>
                                {details !== null ?
                                    <div>
                                        {/* <div id="resetPassword" className="popup fade" role="dialog">
                                            <div className="ico-dialog">
                                                <div className="ico-content">

                                                    <div className="ico-header">
                                                        <button type="button" className="close" data-dismiss="ico" onClick={api.HidePopup.bind(this, 'resetPassword')}>&times;</button>
                                                        <h4 className="ico-title">Reset Password</h4>
                                                    </div>

                                                    <div className="ico-body">
                                                        <label htmlFor="password">Password * :</label>
                                                        <input type="password" ref="password" className="form-control" onChange={() => this._validatePass()} />
                                                        <label htmlFor="confirmPassword">Confirm Password * :</label>
                                                        <input type="password" ref="confirmPassword" className="form-control" onChange={() => this._validatePass()} />
                                                        {errorMgs}
                                                    </div>

                                                    <div className="ico-footer">
                                                        <button type="button" data-id={details.id} className="btn btn-success" onClick={this.updateUserPassword.bind(this)} disabled={!this.state.validate}>Update</button>
                                                    </div>

                                                </div>
                                            </div>
                                        </div> */}
                                        <div className="module-form ">
                                            <div className="form-group">
                                                <label htmlFor="name">Name :</label>
                                                <div className="name-width">
                                                    <input type="text" ref="name" defaultValue={details.username} id="name" className="form-control" name="name" autoFocus onChange={() => this.setState({ error: '' })} />
                                                </div>
                                            </div>

                                            <div className="form-group">
                                                <label htmlFor="email">Email :</label>
                                                <input type="email" ref="email" disabled id="email" defaultValue={details.email} className="form-control" name="email" onChange={() => this.setState({ error: '' })} />
                                            </div>

                                            <div className="form-group">
                                                <label htmlFor="role">Role :</label>
                                                <input type="role" ref="role" defaultValue={details.role} className="form-control" name="role" onChange={() => this.setState({ error: '' })} />
                                            </div>

                                            <div className="form-group access" style={{ marginTop: '20px' }}>
                                                <label htmlFor="addresss">POS Access :</label>
                                                <div style={{ margin: '0px 0px 10px 20px' }}>
                                                    <input id='chkbox' type='checkbox' className="chkbox" checked={this.state.posNetwork} onChange={() => this.onPOSNetworkChange()}></input>
                                                    <label for='chkbox' style={{ margin: 0, marginLeft: 10 }}>Network</label>
                                                </div>
                                            </div>

                                            {this.state.posNetwork ? '' :
                                                <div className="dropdowns">
                                                    <div className="form-group">
                                                        <label htmlFor="country">Region :</label>
                                                        <select className="form-control _country valid" onChange={() => this.callRegion()} id="region" >
                                                            <option value="*">All</option>
                                                            {regionOptionItems}
                                                        </select>
                                                    </div>

                                                    <div className="form-group">
                                                        <label htmlFor="country">Country :</label>
                                                        <select className="form-control _country valid" onChange={() => this.callCountry()} id="country" >
                                                            <option value="*">All</option>
                                                            {countryOptionItems}
                                                        </select>
                                                    </div>

                                                    <div className="form-group">
                                                        <label htmlFor="country">City :</label>
                                                        <select className="form-control _country valid" onChange={(e) => this.callCity(e)} name="region">
                                                            <option value="*">All</option>
                                                            {cityOptionItems}
                                                        </select>
                                                    </div>
                                                </div>}


                                            <div className="form-group access">
                                                <label htmlFor="addresss">Route Access :</label>
                                                <div style={{ margin: '0px 0px 10px 20px' }}>
                                                    <input id='chkboxR' type='checkbox' className="chkboxR" checked={this.state.routeNetwork} onChange={() => this.onRouteNetworkChange()}></input>
                                                    <label for='chkboxR' style={{ margin: 0, marginLeft: 10 }}>Network</label>
                                                </div>
                                            </div>

                                            {this.state.routeNetwork ? '' :
                                                <div className="dropdowns">
                                                    <div className="form-group">
                                                        <label htmlFor="country">Route Group :</label>
                                                        <select className="form-control _country valid" onChange={(e) => this.callRouteGroup(e)} id="routeGroup" >
                                                            <option value="Domestic" selected={this.state.routeGroup === 'Domestic'}>Domestic</option>
                                                            <option value="International" selected={this.state.routeGroup === 'International'}>International</option>
                                                        </select>
                                                    </div>

                                                    <div className="form-group">
                                                        <label htmlFor="country">Route Region :</label>
                                                        <select className="form-control _country valid" onChange={() => this.callRouteRegion()} id="routeRegion" >
                                                            <option value="*">All</option>
                                                            {routeRegions}
                                                        </select>
                                                    </div>

                                                    <div className="form-group">
                                                        <label htmlFor="country">Route Country :</label>
                                                        <select className="form-control _country valid" onChange={() => this.callRouteCountry()} id="routeCountry" >
                                                            <option value="*">All</option>
                                                            {routeCountries}
                                                        </select>
                                                    </div>

                                                    <div className="form-group">
                                                        <label htmlFor="country">Route :</label>
                                                        <select className="form-control _country valid" onChange={(e) => this.callRoute(e)} name="route">
                                                            <option value="*">All</option>
                                                            {routes}
                                                        </select>
                                                    </div>
                                                </div>}

                                            <div className="form-group" style={{ marginTop: '10px', width: '100%' }}>
                                                <label htmlFor="addresss">System Modules :</label>
                                                <div style={{ margin: '10px 0px 10px 0px;', display: 'flex', 'width': '100%', 'flex-wrap': 'wrap' }}>
                                                    {this.state.checkboxArray.map((checkbox, i) => {
                                                        return (
                                                            <div className="edit-user-modules">
                                                                <input
                                                                    checked={this.state.checkboxSelctedCountIndex.includes(checkbox.id)}
                                                                    type="checkbox" value={`${checkbox.name}`}
                                                                    onClick={() => this.onClickCheckbox(checkbox.id)} />
                                                                <label style={{ marginLeft: '10px' }}>{checkbox.name}</label>
                                                            </div>
                                                        )
                                                    })}
                                                </div>
                                            </div>

                                            {errorMgs}
                                        </div>
                                        <div className='action-btn'>
                                            <button type="submit" onClick={this._validate.bind(this)} className="btn">Update User</button>
                                        </div>
                                    </div> : <div />}
                            </div>
                        </div>
                    </div>
                    <Modal
                        show={this.state.displayModal}
                        aria-labelledby="ModalHeader"
                        onHide={() => this.setState({ displayModal: false })}
                    >
                        <Modal.Header >
                            <Modal.Title id='ModalHeader'>Reset Password</Modal.Title>
                        </Modal.Header>
                        <Modal.Body className='resetpass-body'>
                            <label htmlFor="password">Password * :</label>
                            <input type="password" ref="password" className="form-control" onChange={() => this._validatePass()} />
                            <label htmlFor="confirmPassword">Confirm Password * :</label>
                            <input type="password" ref="confirmPassword" className="form-control" onChange={() => this._validatePass()} />
                            {errorMgsPass}
                        </Modal.Body>
                        <Modal.Footer>
                            <button type="button" className="btn btn-success" onClick={this.updateUserPassword.bind(this)} disabled={!this.state.disable}>Update</button>
                        </Modal.Footer>
                    </Modal>
                </div>
            </div>
        );
    }
}

export default EditUser;