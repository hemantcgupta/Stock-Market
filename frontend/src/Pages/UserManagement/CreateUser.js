import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import Access from "../../Constants/accessValidation";
import api from '../../API/api'
import $ from 'jquery';
import './UserManagement.scss';
import _ from 'lodash';
import Swal from 'sweetalert2';


const apiServices = new APIServices();
let routeAccess = {
    selectedRouteGroup: ['Domestic']
};


class CreateUser extends Component {
    constructor(props) {
        super(props);
        this.state = {
            msg: "",
            regions: [],
            regionSelected: 'Null',
            countries: [],
            countrySelected: 'Null',
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
            posNetwork: false,
            posAccess: '',
            domestic: false,
            international: false,
            validate: false,
            checkboxArray: [],
            checkboxSelctedCountIndex: []

        };
        this.hideAlertSuccess = this.hideAlertSuccess.bind(this);

    }

    componentDidMount() {
        if (Access.accessValidation('Users', 'createUser')) {
            this.getRegions();
            this.getRouteRegions();
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

    getRegions = () => {
        api.get(`getRegion`)
            .then((response) => {
                const regions = response.data;
                this.setState({
                    regions: regions,
                });
            })
            .catch((err) => {
                console.log("something went wrong with countries Api");
            })
    }

    getCountries = () => {
        if (this.state.regionSelected !== '*') {
            api.get(`getCountryByRegionId?regionId=${encodeURIComponent(this.state.regionSelected)}`)
                .then((response) => {
                    const countries = response.data;
                    this.setState({
                        countries: countries,
                    });
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
                });
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
                    });
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
        var mailformat = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/;
        var phoneno = /^\d{10}$/;
        if ((this.refs.name.value.trim()) === '') {
            this.setState({
                error: 'Name is required field'
            })
        }
        else if ((this.refs.email.value).trim() === '') {
            this.setState({
                error: 'Email is required field'
            })
        }
        else if (!this.refs.email.value.trim().match(mailformat)) {
            this.setState({
                error: 'Email is Not Valid'
            })
        }
        // else if ((this.refs.password.value).trim() === '') {
        //     this.setState({
        //         error: 'Password is required field'
        //     })
        // }
        // else if ((this.refs.password.value).trim().length < 8) {
        //     this.setState({
        //         error: 'Password should be min 8 characters long'
        //     })
        // }
        // else if ((this.refs.password.value).trim() !== this.refs.confirmPassword.value.trim()) {
        //     this.setState({
        //         error: 'Confirm Password should Match with Password'
        //     })
        // }
        else if ((this.refs.role.value).trim() === '') {
            this.setState({
                error: 'Role is required field'
            })
        }
        else if (this.state.posAccess === '') {
            this.setState({
                error: 'Please select Pos Access'
            })
        } else if (this.state.checkboxSelctedCountIndex.length === 0) {
            this.setState({
                error: 'Please select system module'
            })
        } else {
            this.setState({
                error: '',
                validate: true
            }, () => this.createUser())
        }
    }

    createUser = () => {
        if (this.state.validate) {
            let systemModules = this.state.checkboxSelctedCountIndex.map((chkbox, i) => {
                return this.state.checkboxArray[chkbox].id;
            })
            if (systemModules.length > 0) {
                systemModules = systemModules.join(',');
            } else {
                systemModules = '';
            }
            const name = (this.refs.name.value).trim();
            const email = (this.refs.email.value).trim();
            const password = '1234567890';
            const role = (this.refs.role.value).trim();
            const access = this.state.posAccess;
            const route_access = JSON.stringify(routeAccess);


            var postData = {
                email: email,
                password: password,
                username: name,
                role: role,
                access: access,
                route_access: route_access,
                systemmodule: systemModules
            };

            api.post('api/register/', postData)
                .then((res) => {
                    Swal.fire({
                        title: 'Success!',
                        text: `'User created successfully`,
                        icon: 'success',
                        confirmButtonText: 'Ok'
                    }).then(() => {
                        this.hideAlertSuccess()
                    })

                })
                .catch((err) => {
                    Swal.fire({
                        title: 'Error!',
                        text: err.response.data.email[0],
                        icon: 'error',
                        confirmButtonText: 'Ok'
                    })
                })
        }
    }


    hideAlertSuccess() {
        this.props.history.push(`/searchUser`)
    }

    hideAlertFailure() {
        $('input').val('');
    }

    onPOSNetworkChange() {
        this.setState({ posNetwork: !this.state.posNetwork, error: '' }, () => this.setPOSNetworkLevel())
    }

    setPOSNetworkLevel() {
        if (this.state.posNetwork) {
            this.setState({ posAccess: '#*', regions: [], countries: [], cities: [] })
        }
        else {
            this.getRegions();
            this.setState({ posAccess: '' })
        }
    }

    onRouteNetworkChange() {
        this.setState({ routeNetwork: !this.state.routeNetwork, error: '' }, () => this.setRouteNetworkLevel())
    }

    setRouteNetworkLevel() {
        if (this.state.routeNetwork) {
            routeAccess = {}
            this.setState({ routeRegions: [], routeCountries: [], routes: [] })
        }
        else {
            this.setState({ routeGroup: 'Domestic' }, () => this.getRouteRegions())
            routeAccess = { selectedRouteGroup: ['Domestic'] }
        }
    }


    onClickCheckbox(checbox, index) {
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

    render() {
        let regionOptionItems = this.state.regions.map((region) =>
            <option value={region.Region}>{region.Region}</option>
        );
        let countryOptionItems = this.state.countries.map((country) =>
            <option value={country.CountryCode}>{country.CountryCode}</option>
        );
        let cityOptionItems = this.state.cities.map((city) =>
            <option value={city.CityCode}>{city.CityCode}</option>
        );
        let routeRegions = this.state.routeRegions.map((region) =>
            <option value={region.Route_Region}>{region.Route_Region}</option>
        );
        let routeCountries = this.state.routeCountries.map((country) =>
            <option value={country.Route_Country}>{country.Route_Country}</option>
        );
        let routes = this.state.routes.map((route) =>
            <option value={route.Route}>{route.Route}</option>
        );
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
                                    <h2>Create User</h2>
                                    <div className="clearfix"></div>
                                </div>
                                <div>
                                    <div className="module-form ">
                                        <div className="form-group">
                                            <label htmlFor="name">Name :</label>
                                            <div className="name-width">
                                                <input type="text" ref="name" id="name" className="form-control" name="name" autoFocus onChange={() => this.setState({ error: '' })} />
                                            </div>
                                        </div>

                                        <div className="form-group">
                                            <label htmlFor="email">Email :</label>
                                            <input type="email" ref="email" id="email" className="form-control" name="email" onChange={() => this.setState({ error: '' })} />
                                        </div>

                                        {/* <div className="form-group">
                                            <label htmlFor="password">Password * :</label>
                                            <input type="password" ref="password" className="form-control" name="password" onChange={() => this.setState({ error: '' })} />
                                        </div>

                                        <div className="form-group">
                                            <label htmlFor="confirmPassword">Confirm Password * :</label>
                                            <input type="password" ref="confirmPassword" className="form-control" name="confirmPassword" onChange={() => this.setState({ error: '' })} />
                                        </div> */}

                                        <div className="form-group">
                                            <label htmlFor="role">Role :</label>
                                            <input type="role" ref="role" className="form-control" name="role" onChange={() => this.setState({ error: '' })} />
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
                                                        <option value="Domestic">Domestic</option>
                                                        <option value="International">International</option>
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

                                        <div className="form-group system-module-list">
                                            <label htmlFor="addresss">System Modules :</label>
                                            <div style={{ margin: '10px 0px 10px 0px;', display: 'flex', 'width': '100%', 'flex-wrap': 'wrap' }}>
                                                {this.state.checkboxArray.map((checkbox, i) => {
                                                    return (
                                                        <div className='edit-user-modules'>
                                                            <input checked={this.state.checkboxSelctedCountIndex.includes(i)} type="checkbox" value={`${checkbox.name}`} onClick={() => this.onClickCheckbox(checkbox, i)} />
                                                            <label style={{ marginLeft: '10px' }}>{checkbox.name}</label>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        </div>

                                        {errorMgs}
                                    </div>
                                    <div className='action-btn'>
                                        <button type="submit" onClick={this._validate.bind(this)} className="btn">Create User</button>
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

export default CreateUser;